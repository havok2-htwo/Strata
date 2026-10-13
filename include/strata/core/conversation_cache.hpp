// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // STRATA_SLOT_PARK: the turn checkpoint a batch slot's conversation was parked with (its own, not one of the chain
    // it was admitted with).  Once that conversation goes on, the next slot image of it does not carry this one along
    // (it holds the new turn's).  A run-time mark only, not in the session file.
    bool slot_turn = false;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            part.slot_turn = c.slot_turn;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

/// What conversation_checkpoints_split(checks, stages) puts in stage `k`'s list (0: `stage0`, k >= 1: `parts[k - 1]`),
/// COPIED, so `checks` stays as it is: a capture that runs beside the serve loop (STRATA_PARK_ASYNC) reads the chain
/// while the loop may read it too.  Without a split (stages == 0) every checkpoint as it is.  `without_slot_turn`:
/// the ones marked slot_turn are left out (STRATA_SLOT_PARK: the chain a batch slot's image takes along).
inline std::vector<ConversationCheckpoint> conversation_checkpoints_stage_copy(
        const std::vector<ConversationCheckpoint>& checks, size_t stages, size_t k, bool without_slot_turn = false) {
    std::vector<ConversationCheckpoint> out;
    if (stages == 0) {
        if (!without_slot_turn) {
            out = checks;
            return out;
        }
        size_t n = 0;
        for (const auto& c : checks) n += c.slot_turn ? 0 : 1;
        out.reserve(n);
        for (const auto& c : checks)
            if (!c.slot_turn) out.push_back(c);
        return out;
    }
    if (k > stages) return out;
    size_t whole = 0;
    for (const auto& c : checks) whole += c.stage_parts.size() == stages && !(without_slot_turn && c.slot_turn) ? 1 : 0;
    out.reserve(whole);
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages || (without_slot_turn && c.slot_turn)) continue;
        const ConversationCheckpoint& state = k == 0 ? c : c.stage_parts[k - 1];
        ConversationCheckpoint part;
        part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned; part.slot_turn = c.slot_turn;
        part.gdn = state.gdn; part.ple = state.ple; part.tails = state.tails; part.dead = state.dead;
        part.block_pos = state.block_pos;
        out.push_back(std::move(part));
    }
    return out;
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;
    /// The engine runs a drafter, but this image holds no draft-layer K/V: a batch slot's conversation
    /// (STRATA_SLOT_PARK; slots in pipelined groups draft nothing).  It is validated and restored without the draft
    /// layer, whose K/V below it then stays as it was - as after a slot's give-back: drafts may differ, tokens never.
    /// A run-time mark only, not in a session file.
    bool draftless = false;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

/// Every byte, id and mark of two checkpoints alike (stage parts too); vector capacities aside.
inline bool conversation_checkpoint_same(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.gdn != b.gdn || a.ple != b.ple || a.tails != b.tails ||
        a.dead != b.dead || a.block_pos != b.block_pos || a.used != b.used || a.pinned != b.pinned ||
        a.slot_turn != b.slot_turn ||
        a.stage_parts.size() != b.stage_parts.size()) return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!conversation_checkpoint_same(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}
/// A dropped checkpoint made ready for reuse (STRATA_CKPT_POOL): its tokens, pictures and every run-time mark
/// cleared (stage parts too), its running-state vectors kept for the next save to write into.  slot_turn is cleared
/// too: a recycled slot-turn checkpoint taken for a main-session save was otherwise left out of slot chains.
inline void conversation_checkpoint_clear_marks(ConversationCheckpoint& c) {
    c.ids.clear(); c.imgs.clear(); c.used = 0; c.pinned = false; c.slot_turn = false;
    for (ConversationCheckpoint& part : c.stage_parts) conversation_checkpoint_clear_marks(part);
}
inline bool conversation_kv_same(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim &&
           a.k == b.k && a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}
/// Two parked images alike in every byte (STRATA_PARK_ASYNC_VERIFY: a capture against another of the same state).
inline bool conversation_image_same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        a.draftless != b.draftless || !conversation_checkpoint_same(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() ||
        a.kv.size() != b.kv.size() || a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!conversation_checkpoint_same(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!conversation_kv_same(a.kv[i], b.kv[i])) return false;
    for (size_t i = 0; i < a.stage_images.size(); ++i)
        if (!conversation_image_same(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    /// Puts back what take_reuse() returned when nothing used it (a capture that did not start): the same pages and
    /// extents, as if they had never been taken.
    void restore_reuse(ConversationKvReuse&& reuse) { reuse_ = std::move(reuse); }
    size_t budget() const { return budget_; }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        return out;
    }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the caller skips parking it - the pinned prefix is what the queries come back to)
            if (!evict_oldest(true)) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest.  This is make_room()'s loop body, so
    // the parking path can also free RAM one entry at a time on demand (see the physical-RAM admission
    // gate in generate.cpp).  A ConversationBuffer is a list of 16 MiB segments and each segment is its
    // own allocation, far above glibc's mmap threshold, so dropping an entry returns the whole footprint
    // to the kernel at once - the next admission check reads it back from /proc/meminfo.
    // False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    // `spill` (make_room's evictions: the cache made room for another image): an eviction sink may keep the image
    // (STRATA_CONV_TIER_DIR, conversation_tier.hpp).  The physical-RAM gate and a save's reclaim evict to get the memory
    // back, so their victims are freed here at once - the sink only hears of them.
    bool evict_oldest(bool spill = false) {
        auto victim = std::find_if(entries_.begin(), entries_.end(), [](const SavedConversation& e) { return !e.pinned(); });
        if (victim == entries_.end()) return false;
        bytes_ -= victim->bytes();
        ++evictions_;
        if (!evict_sink_) {
            entries_.erase(victim);
            return true;
        }
        SavedConversation image = std::move(*victim);
        entries_.erase(victim);
        evict_sink_(std::move(image), spill);
        return true;   // (whatever the sink left of `image` is freed here)
    }
    /// Where evicted images go (none: freed).  Never drop_superseded's or clear()'s: those are not wanted any more.
    using EvictSink = std::function<void(SavedConversation&&, bool spill)>;
    void set_evict_sink(EvictSink sink) { evict_sink_ = std::move(sink); }

    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        entries_.clear();
        bytes_ = 0;
        reuse_ = {};
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    //
    // set_supersede_prefix(true) (STRATA_SUPERSEDE_PREFIX, implied by STRATA_SLOT_PARK) decides by tokens instead.
    // With STRATA_SLOT_CKPT_MOVE a turn checkpoint moves into its batch slot, so a parked chain often holds only the
    // system prompt's root, and the rule above then drops every other conversation that shares that root; a batch
    // slot's image (STRATA_SLOT_PARK) holds only its turn checkpoint, so the chain rule never finds its predecessor.
    // By prefix, an entry that holds no pinned prefix is superseded when its live tokens start the outgoing ones
    // (this conversation, earlier), or when its deepest checkpoint starts them and is longer than the outgoing chain's
    // shallowest checkpoint (deeper than any root the two may share).  A root alone never drops an entry.
    void set_supersede_prefix(bool on) { supersede_prefix_ = on; }
    bool supersede_prefix() const { return supersede_prefix_; }
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        if (supersede_prefix_) return drop_superseded_by_prefix(ids, images, checkpoints, cvec);
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
                bytes_ -= e.bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }
    size_t drop_superseded_by_prefix(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                     const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        // c's tokens, and the pictures among them, start the outgoing conversation
        auto starts = [&](const ConversationCheckpoint& c) {
            const size_t n = c.ids.size();
            if (n == 0 || n > ids.size() || !std::equal(c.ids.begin(), c.ids.end(), ids.begin())) return false;
            size_t j = 0;
            for (const auto& image : images) {
                if (image.start >= (int64_t) n) continue;
                if (j == c.imgs.size() || !(c.imgs[j++] == image)) return false;
            }
            return j == c.imgs.size();
        };
        size_t root = 0;   // the outgoing chain's shallowest checkpoint (0: none - then only the live rule applies)
        for (const auto& k : checkpoints)
            if (!k.ids.empty() && (root == 0 || k.ids.size() < root)) root = k.ids.size();
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            const bool earlier = starts(e.live) ||
                                 (deepest && root > 0 && deepest->ids.size() > root && starts(*deepest));
            if (e.cvec == cvec && !e.pinned() && earlier) {
                bytes_ -= e.bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        bytes_ += n;
        return true;
    }

private:
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0;
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    bool supersede_prefix_ = false;   // drop_superseded by tokens (set_supersede_prefix)
    EvictSink evict_sink_;            // set_evict_sink
    std::deque<SavedConversation> entries_; // least recently active first
    ConversationKvReuse reuse_;
};

} // namespace strata::core
