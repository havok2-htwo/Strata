// CPU-only tests for the conversation cache's disk tier (include/strata/core/conversation_tier.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_tier.hpp"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}
std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}
std::vector<int32_t> tokens(size_t n, int32_t base) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + int32_t(i % 977);
    return v;
}
ConversationCheckpoint state(const std::vector<int32_t>& ids, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 40 + seed;
    return c;
}
// one stage's image: the live state, a root checkpoint (pinned mark) and a turn checkpoint (slot_turn mark), 2 layers
SavedConversation part(const std::vector<int32_t>& ids, int64_t lo, int64_t hi, uint8_t seed, bool pin = false) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.cvec = true;
    s.live = state(ids, seed);
    s.checkpoints.push_back(state(std::vector<int32_t>(ids.begin(), ids.begin() + 50), seed + 10));
    s.checkpoints.back().pinned = pin;
    s.checkpoints.push_back(state(std::vector<int32_t>(ids.begin(), ids.end() - 20), seed + 20));
    s.checkpoints.back().slot_turn = true;
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = (int64_t) ids.size(); kv.heads = 2; kv.head_dim = 64; kv.page_size = 64;
        kv.pooled_rows = 4; kv.idx_dim = 16;
        kv.k = pattern(ids.size() * 8 + layer, uint8_t(seed + layer));
        kv.v = pattern(ids.size() * 8, uint8_t(seed + layer + 3));
        kv.k_scale = pattern(64, uint8_t(seed + 5));
        kv.v_scale = pattern(64, uint8_t(seed + 6));
        kv.pooled = pattern(4 * 16, uint8_t(seed + 7));
        s.kv.push_back(std::move(kv));
    }
    return s;
}
// a layer split's image: stage 0 plus three later stages (the first marked like a batch slot's: draftless)
SavedConversation image_of(const std::vector<int32_t>& ids, uint8_t seed, bool pin = false) {
    SavedConversation s = part(ids, 0, 12, seed, pin);
    s.draftless = true;
    for (int k = 1; k < 4; ++k) s.stage_images.push_back(part(ids, 12 * k, 12 * k + 12, uint8_t(seed + 30 * k), pin));
    return s;
}
SavedConversation image(size_t n, int32_t base = 1000, uint8_t seed = 1, bool pin = false) {
    return image_of(tokens(n, base), seed, pin);
}
std::vector<int32_t> extend(const std::vector<int32_t>& ids, size_t more) {
    std::vector<int32_t> v = ids;
    v.push_back(1);
    for (size_t i = 0; i < more; ++i) v.push_back(int32_t(5000 + i));
    return v;
}

struct Fixture {
    std::string dir;
    int64_t t = 1'760'000'000;
    std::vector<std::string> lines;
    std::mutex mu;
    explicit Fixture(const char* name) {
        dir = (fs::temp_directory_path() / ("strata_tier_test_" + std::to_string((unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + name)).string();
        fs::remove_all(dir);
    }
    ~Fixture() { std::error_code ec; fs::remove_all(dir, ec); }
    ConversationTierConfig config() {
        ConversationTierConfig c;
        c.dir = dir;
        c.min_tokens = 100;
        c.budget_bytes = 1ull << 30;
        c.cap_bytes = 1ull << 30;
        c.max_images = 100;
        c.ttl_s = 3600;
        c.inflight_bytes = 1ull << 30;
        c.regrow_tokens = 64;
        c.min_free_bytes = 0;
        c.durable = false;
        c.id = {0x1111, 0x2222};
        c.clock = [this] { return t; };
        c.log = [this](const std::string& l) { std::lock_guard<std::mutex> lk(mu); lines.push_back(l); };
        return c;
    }
    size_t files() const {
        size_t n = 0;
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
            if (it->path().filename() != "budget") ++n;
        return n;
    }
    bool any_temp() const {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string n = it->path().filename().string();
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) return true;
        }
        return false;
    }
};

bool start_ok(ConversationTier& t, std::string& err) {
    const bool ok = t.start(err);
    if (!ok) std::fprintf(stderr, "start: %s\n", err.c_str());
    return ok;
}

// read limits as the engine passes them: open (the engine's are its runtime's); one per part
std::vector<SessionReadLimits> limits(size_t parts) { return std::vector<SessionReadLimits>(parts); }

std::vector<int32_t> prompt_after(const SavedConversation& s, size_t more = 10) { return extend(s.live.ids, more); }

void test_spill_only_on_eviction_and_restore_exact() {
    Fixture f("evict");
    ConversationTier tier(f.config());
    std::string err;
    check(start_ok(tier, err), "the tier starts");
    ConversationCache cache(1ull << 30, 2);
    cache.set_evict_sink([&](SavedConversation&& img, bool spill) { tier.evicted(std::move(img), spill); });
    const SavedConversation a = image(300, 1000, 1), b = image(310, 2000, 2), c = image(320, 3000, 3);
    check(cache.put(SavedConversation(a)) && cache.put(SavedConversation(b)), "two images parked");
    tier.drain();
    check(tier.stats().spills == 0 && f.files() == 0, "nothing is written while the RAM cache holds everything");
    check(cache.put(SavedConversation(c)), "a third image: make_room evicts the oldest");
    tier.drain();
    auto st = tier.stats();
    check(st.spills == 1 && st.files == 1 && st.parts == 4 && f.files() == 5, "the evicted image went to disk: 4 parts + meta");
    check(st.written_today == st.disk_bytes && st.written_bytes == st.disk_bytes && st.disk_bytes > 0, "bytes counted");
    // the physical-RAM gate's eviction: freed now, counted, never written
    check(cache.evict_oldest() && cache.size() == 1, "the gate evicts one");
    tier.drain();
    st = tier.stats();
    check(st.spills == 1 && st.drops_ram == 1 && f.files() == 5, "a gate eviction is not written (drops_ram)");
    // drop_superseded and clear() never reach the sink
    cache.set_supersede_prefix(true);
    cache.drop_superseded(prompt_after(c), {}, {}, true);
    cache.clear();
    tier.drain();
    st = tier.stats();
    check(st.spills == 1 && st.drops_ram == 1 && st.drops_small == 0, "superseded and cleared images are not spilled");
    // the restore: the image that was evicted, byte for byte (stage images and marks included)
    const auto m = tier.best(prompt_after(a), {}, true);
    check(m.key != 0 && m.tokens == 300 && m.live && m.parts == 4 && m.draftless, "best finds the spilled conversation");
    check(tier.best(prompt_after(b), {}, true).tokens == 0, "the gate-evicted one is not on disk");
    check(tier.best(prompt_after(a), {}, false).tokens == 0, "another control vector mode does not match");
    const auto ck = tier.best(extend(std::vector<int32_t>(a.live.ids.begin(), a.live.ids.end() - 20), 5), {}, true);
    check(ck.tokens == 280 && !ck.live, "a checkpoint matches too (a rewritten tail)");
    SavedConversation back;
    ConversationTier::LoadInfo info;
    check(tier.load(m.key, limits(4), back, err, &info), "load");
    check(!info.from_queue && info.bytes > 0, "read from disk");
    check(conversation_image_same(back, a), "the restored image is byte-identical to the evicted one (marks too)");
    check(tier.stats().restores == 1 && tier.stats().read_bytes == info.bytes, "restore counted");
    check(conversation_image_digest(back) == conversation_image_digest(a), "digest equal");
    SavedConversation other = a;
    other.stage_images[2].checkpoints[1].slot_turn = false;
    check(conversation_image_digest(other) != conversation_image_digest(a), "the digest sees a mark");
}

void test_min_tokens_and_write_once() {
    Fixture f("once");
    ConversationTier tier(f.config());
    std::string err;
    check(start_ok(tier, err), "start");
    tier.evicted(image(99), true);
    tier.drain();
    check(tier.stats().drops_small == 1 && tier.stats().spills == 0 && f.files() == 0, "below min_tokens: dropped");
    const SavedConversation a = image(400, 1000, 1);
    tier.evicted(SavedConversation(a), true);
    tier.drain();
    const auto written = tier.stats().written_bytes;
    check(tier.stats().spills == 1 && written > 0, "written once");
    // restored and evicted again unchanged: nothing written
    tier.evicted(SavedConversation(a), true);
    tier.drain();
    check(tier.stats().spills == 1 && tier.stats().kept == 1 && tier.stats().written_bytes == written,
          "an unchanged conversation is never written twice");
    // the same conversation a little further on (fewer than regrow_tokens new): the file holds it
    const auto grown = extend(a.live.ids, 40);
    SavedConversation g1 = part(grown, 0, 12, 7);
    tier.evicted(std::move(g1), true);
    tier.drain();
    check(tier.stats().spills == 1 && tier.stats().kept == 2, "grown by less than regrow_tokens: kept, not rewritten");
    // an older, shorter copy of it (a turn back): held by the file of the later state
    SavedConversation older = part(std::vector<int32_t>(a.live.ids.begin(), a.live.ids.begin() + 200), 0, 12, 8);
    tier.evicted(std::move(older), true);
    tier.drain();
    check(tier.stats().spills == 1 && tier.stats().kept == 3, "an earlier copy of a conversation on disk is not written");
    // grown by regrow_tokens or more: written in full, the older file goes
    tier.evicted(image_of(extend(a.live.ids, 100), 9), true);
    tier.drain();
    const auto st = tier.stats();
    check(st.spills == 2 && st.superseded == 1 && st.files == 1 && f.files() == 5,
          "grown past regrow_tokens: rewritten, the older file set removed");
    check(tier.best(extend(extend(a.live.ids, 100), 3), {}, true).tokens == 501, "matched at its new length");
}

void test_budget_and_day() {
    Fixture f("budget");
    auto cfg = f.config();
    const uint64_t one = image(300).bytes();
    cfg.budget_bytes = one + one / 2;   // room for one image a day
    ConversationTier tier(cfg);
    std::string err;
    check(start_ok(tier, err), "start");
    tier.evicted(image(300, 1000), true);
    tier.drain();
    tier.evicted(image(300, 2000), true);
    tier.drain();
    auto st = tier.stats();
    check(st.spills == 1 && st.drops_budget == 1 && st.files == 1, "the second write would pass the budget: dropped");
    check(st.budget == cfg.budget_bytes && st.budget_left == cfg.budget_bytes - st.written_today, "budget left");
    f.t += 86400;   // the next day
    tier.evicted(image(300, 3000), true);
    tier.drain();
    st = tier.stats();
    check(st.spills == 2 && st.drops_budget == 1, "a new day: the budget starts again");
    auto cfg0 = f.config();
    cfg0.budget_bytes = 0;
    Fixture g("budget0");
    cfg0.dir = g.dir;
    ConversationTier none(cfg0);
    check(start_ok(none, err), "start");
    none.evicted(image(300), true);
    none.drain();
    check(none.stats().drops_budget == 1 && g.files() == 0, "budget 0: nothing is ever written");
}

void test_budget_persists() {
    Fixture f("budgetfile");
    auto cfg = f.config();
    {
        ConversationTier tier(cfg);
        std::string err;
        check(start_ok(tier, err), "start");
        tier.evicted(image(300), true);
        tier.drain();
    }
    ConversationTier tier(cfg);
    std::string err;
    check(start_ok(tier, err), "start again");
    check(tier.stats().written_today > 0, "the day's written bytes survive a restart");
    check(tier.stats().files == 0 && f.files() == 0, "not warm: the earlier run's files are removed at start");
}

void test_ttl_and_cap() {
    Fixture f("ttl");
    auto cfg = f.config();
    const uint64_t one = image(300).bytes();
    cfg.ttl_s = 3600;
    cfg.max_images = 2;
    ConversationTier tier(cfg);
    std::string err;
    check(start_ok(tier, err), "start");
    tier.evicted(image(300, 1000), true);
    tier.drain();
    f.t += 10;
    tier.evicted(image(300, 2000), true);
    tier.drain();
    f.t += 10;
    // touch the first (a restore): the second is now the least recently used
    SavedConversation back;
    check(tier.load(tier.best(prompt_after(image(300, 1000)), {}, true).key, limits(4), back, err), "load the first");
    f.t += 10;
    tier.evicted(image(300, 3000), true);
    tier.drain();
    auto st = tier.stats();
    check(st.files == 2 && st.evicted == 1, "max_images: the least recently used set goes");
    check(tier.best(prompt_after(image(300, 2000)), {}, true).tokens == 0, "it was the unused one");
    check(tier.best(prompt_after(image(300, 1000)), {}, true).tokens == 300, "the restored one stays");
    // the byte cap
    Fixture g("cap");
    auto c2 = g.config();
    c2.cap_bytes = one * 2;   // two images' RAM; a set on disk is about that size
    ConversationTier capped(c2);
    check(start_ok(capped, err), "start");
    for (int i = 0; i < 4; ++i) { capped.evicted(image(300, 1000 * (i + 1)), true); capped.drain(); g.t += 5; }
    st = capped.stats();
    check(st.disk_bytes <= c2.cap_bytes && st.evicted >= 2 && st.spills == 4, "cap_bytes holds, LRU evicted");
    // the time to live: an hour without use, then the next write's sweep removes it
    f.t += 7200;
    tier.evicted(image(300, 9000), true);
    tier.drain();
    st = tier.stats();
    check(st.expired == 2 && st.files == 1, "expired by last use");
}

// the writer held at a write: the queue, the in-flight bound, a restore from the queue, a restore that waits
void test_queue_and_inflight() {
    Fixture f("queue");
    auto cfg = f.config();
    std::mutex mu;
    std::condition_variable cv;
    bool hold = true;
    std::atomic<int> entered{0};
    cfg.fault = [&](const char* step) {
        if (std::strcmp(step, "write") == 0) {
            ++entered;
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !hold; });
        }
        return 0;
    };
    const uint64_t one = image(300).bytes();
    cfg.inflight_bytes = one + one / 2;
    ConversationTier tier(cfg);
    std::string err;
    check(start_ok(tier, err), "start");
    const SavedConversation a = image(300, 1000), b = image(300, 2000), c = image(300, 3000);
    tier.evicted(SavedConversation(a), true);
    while (entered.load() == 0) std::this_thread::yield();   // a is being written (held)
    tier.evicted(SavedConversation(b), true);                // waits
    tier.evicted(SavedConversation(c), true);                // the in-flight bound
    auto st = tier.stats();
    check(st.pending == 1 && st.drops_busy == 1, "the in-flight bound drops what does not fit beside the queue");
    SavedConversation back;
    ConversationTier::LoadInfo info;
    check(tier.load(tier.best(prompt_after(b), {}, true).key, limits(4), back, err, &info), "load from the queue");
    check(info.from_queue && conversation_image_same(back, b), "a waiting image comes back as it is, no read");
    check(tier.stats().pending == 0 && tier.stats().restores_queue == 1, "taken out of the queue");
    // a restore of the one being written waits for the write and then reads it
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        { std::lock_guard<std::mutex> lk(mu); hold = false; }
        cv.notify_all();
    });
    check(tier.load(tier.best(prompt_after(a), {}, true).key, limits(4), back, err, &info), "load waits for the write");
    release.join();
    check(!info.from_queue && conversation_image_same(back, a), "then reads it from disk, byte-identical");
}

void test_io_errors_and_crash_safety() {
    std::string err;
    for (const char* step : {"write", "file_flush", "rename", "meta"}) {
        Fixture f(step);
        auto cfg = f.config();
        cfg.durable = true;
        cfg.fault = [step](const char* s) { return std::strcmp(s, step) == 0 ? EIO : 0; };
        ConversationTier tier(cfg);
        check(start_ok(tier, err), "start");
        SavedConversation img = image(300);
        img.kv[0].k = pattern((17u << 20) + 3, 9);   // a block write happens before the last block (the "write" step)
        tier.evicted(std::move(img), true);
        tier.drain();
        const auto st = tier.stats();
        check(st.drops_io == 1 && st.spills == 0 && st.files == 0, "a failed write is dropped and counted");
        check(f.files() == 0 && !f.any_temp(), "no part, meta or temporary file is left behind");
        check(tier.best(prompt_after(image(300)), {}, true).tokens == 0, "nothing to restore");
    }
    // read errors: the fault, a corrupt part, a missing part -> false, a miss, the files go; never a partial image
    {
        Fixture f("readfault");
        auto cfg = f.config();
        bool fail_read = false;
        cfg.fault = [&](const char* s) { return fail_read && std::strcmp(s, "read") == 0 ? EIO : 0; };
        ConversationTier tier(cfg);
        check(start_ok(tier, err), "start");
        tier.evicted(image(300), true);
        tier.drain();
        fail_read = true;
        SavedConversation back;
        back.layer_lo = 77;
        check(!tier.load(tier.best(prompt_after(image(300)), {}, true).key, limits(4), back, err), "an I/O error: false");
        check(back.layer_lo == 77 && back.kv.empty(), "the output is untouched");
        check(tier.stats().misses == 1 && tier.stats().files == 0 && f.files() == 0, "a miss; the set goes");
    }
    {
        Fixture f("corrupt");
        ConversationTier tier(f.config());
        check(start_ok(tier, err), "start");
        tier.evicted(image(300), true);
        tier.drain();
        const auto m = tier.best(prompt_after(image(300)), {}, true);
        std::string p;
        for (const auto& e : fs::directory_iterator(f.dir))
            if (e.path().string().size() > 6 && e.path().string().compare(e.path().string().size() - 6, 6, ".2.ses") == 0)
                p = e.path().string();
        check(!p.empty(), "part 2 is on disk");
        {
            std::fstream fh(p, std::ios::in | std::ios::out | std::ios::binary);
            fh.seekp(200);
            char x = 0;
            fh.read(&x, 1);
            fh.seekp(200);
            x ^= 0x5a;
            fh.write(&x, 1);
        }
        SavedConversation back;
        check(!tier.load(m.key, limits(4), back, err), "a corrupt part is refused");
        check(err.find("checksum") != std::string::npos, "by its payload checksum");
        check(tier.stats().misses == 1 && f.files() == 0, "the set goes");
    }
    {
        Fixture f("missing");
        ConversationTier tier(f.config());
        check(start_ok(tier, err), "start");
        tier.evicted(image(300), true);
        tier.drain();
        const auto m = tier.best(prompt_after(image(300)), {}, true);
        for (const auto& e : fs::directory_iterator(f.dir))
            if (e.path().string().find(".3.ses") != std::string::npos) fs::remove(e.path());
        SavedConversation back;
        check(!tier.load(m.key, limits(4), back, err) && tier.stats().misses == 1, "a missing part: a miss");
    }
    // RAM refusal (the read preflight): false, but the files stay
    {
        Fixture f("ramref");
        ConversationTier tier(f.config());
        check(start_ok(tier, err), "start");
        tier.evicted(image(300), true);
        tier.drain();
        auto l = limits(4);
        l[1].admit = [](uint64_t, std::string& why) { why = "no RAM"; return false; };
        SavedConversation back;
        check(!tier.load(tier.best(prompt_after(image(300)), {}, true).key, l, back, err), "a RAM refusal");
        check(tier.stats().files == 1 && f.files() == 5, "the files stay for a later try");
        check(tier.load(tier.best(prompt_after(image(300)), {}, true).key, limits(4), back, err), "and load later");
    }
    // a crash between the parts and the commit record, or during a write: at the next start nothing of it is used
    {
        Fixture f("crash");
        auto cfg = f.config();
        {
            ConversationTier tier(cfg);
            check(start_ok(tier, err), "start");
            tier.evicted(image(300), true);
            tier.drain();
        }
        std::string meta;
        for (const auto& e : fs::directory_iterator(f.dir))
            if (e.path().extension() == ".meta") meta = e.path().string();
        fs::remove(meta);   // the commit record never made it
        std::ofstream(f.dir + "/.0123456789abcdef.0.ses.deadbeefdeadbeef.tmp") << "partial";
        cfg.warm = true;
        ConversationTier tier(cfg);
        check(start_ok(tier, err), "start warm");
        check(tier.stats().files == 0 && f.files() == 0, "parts without a commit record and temporary files are removed");
    }
}

void test_warm_identity_and_verify() {
    Fixture f("warm");
    auto cfg = f.config();
    cfg.verify = true;
    const SavedConversation a = image(300, 1000, 1, true);   // marks: pinned, slot_turn, draftless
    std::string err;
    {
        ConversationTier tier(cfg);
        check(start_ok(tier, err), "start");
        tier.evicted(SavedConversation(a), true);
        tier.drain();
        check(tier.stats().verified == 1 && tier.stats().spills == 1, "verify: the write was read back and compared");
    }
    // another identity (another build, model or setting): not indexed, never restored, kept until its time to live
    {
        auto other = cfg;
        other.warm = true;
        other.id.config ^= 1;
        ConversationTier tier(other);
        check(start_ok(tier, err), "start with another identity");
        check(tier.best(prompt_after(a), {}, true).tokens == 0 && tier.stats().foreign == 1 && f.files() == 5,
              "a file of another identity is refused (not indexed) and left alone");
    }
    // the same identity, warm: indexed, restored byte for byte, digest checked
    {
        auto same = cfg;
        same.warm = true;
        ConversationTier tier(same);
        check(start_ok(tier, err), "start warm");
        const auto m = tier.best(prompt_after(a), {}, true);
        check(m.tokens == 300 && tier.stats().files == 1, "a warm start indexes this engine's files");
        SavedConversation back;
        check(tier.load(m.key, limits(4), back, err), "warm load");
        check(conversation_image_same(back, a) && tier.stats().verified == 1, "byte-identical, digest verified");
    }
    // a part file with another identity in its header (swapped in under the same name): refused by the read
    {
        Fixture g("swap");
        auto c1 = g.config();
        auto c2 = g.config();
        c2.id.model ^= 7;
        Fixture h("swapsrc");
        c2.dir = h.dir;
        ConversationTier t1(c1), t2(c2);
        check(start_ok(t1, err) && start_ok(t2, err), "start both");
        t1.evicted(SavedConversation(a), true);
        t2.evicted(SavedConversation(a), true);
        t1.drain();
        t2.drain();
        std::string p1, p2;
        for (const auto& e : fs::directory_iterator(g.dir)) if (e.path().string().find(".1.ses") != std::string::npos) p1 = e.path().string();
        for (const auto& e : fs::directory_iterator(h.dir)) if (e.path().string().find(".1.ses") != std::string::npos) p2 = e.path().string();
        fs::copy_file(p2, p1, fs::copy_options::overwrite_existing);
        SavedConversation back;
        check(!t1.load(t1.best(prompt_after(a), {}, true).key, limits(4), back, err), "a part of another model is refused");
        check(err.find("another model") != std::string::npos && t1.stats().misses == 1, "by its identity");
    }
}

void test_clear_and_reject() {
    Fixture f("clear");
    ConversationTier tier(f.config());
    std::string err;
    check(start_ok(tier, err), "start");
    tier.evicted(image(300, 1000), true);
    tier.evicted(image(300, 2000), true);
    tier.drain();
    check(tier.stats().files == 2, "two on disk");
    tier.reject(tier.best(prompt_after(image(300, 1000)), {}, true).key, "test");
    check(tier.stats().files == 1 && tier.stats().misses == 1, "a refused image goes");
    tier.clear();
    tier.drain();
    check(tier.stats().files == 0 && f.files() == 0, "#879: everything goes");
}

void test_fingerprint() {
    Fixture f("fp");
    fs::create_directories(f.dir);
    const std::string exe = f.dir + "/engine";
    std::ofstream(exe) << "binary one";
    uint64_t a = 0, b = 0, c = 0, d = 0;
    std::string err;
    check(conversation_tier_fingerprint(1, exe, {"--x", "1"}, a, err), "fingerprint");
    check(conversation_tier_fingerprint(1, exe, {"--x", "2"}, b, err) && a != b, "the arguments count");
    check(conversation_tier_fingerprint(2, exe, {"--x", "1"}, c, err) && a != c, "the configuration counts");
    std::ofstream(exe) << "binary two";
    check(conversation_tier_fingerprint(1, exe, {"--x", "1"}, d, err) && a != d, "the binary counts");
    set_env("STRATA_TIER_TEST_SWITCH", "1");
    std::ofstream(exe) << "binary one";
    uint64_t e = 0, g = 0;
    check(conversation_tier_fingerprint(1, exe, {"--x", "1"}, e, err) && a != e, "a STRATA_* variable counts");
    set_env("STRATA_TIER_TEST_SWITCH", nullptr);
    set_env("STRATA_CONV_TIER_TTL_H", "5");
    check(conversation_tier_fingerprint(1, exe, {"--x", "1"}, g, err) && a == g, "the tier's own knobs do not");
    set_env("STRATA_CONV_TIER_TTL_H", nullptr);
}
} // namespace

int main() {
    test_spill_only_on_eviction_and_restore_exact();
    test_min_tokens_and_write_once();
    test_budget_and_day();
    test_budget_persists();
    test_ttl_and_cap();
    test_queue_and_inflight();
    test_io_errors_and_crash_safety();
    test_warm_identity_and_verify();
    test_clear_and_reject();
    test_fingerprint();
    std::printf("conversation_tier_test: %d checks passed\n", checks);
    return 0;
}
