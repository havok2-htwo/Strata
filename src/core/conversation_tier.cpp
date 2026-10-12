#include "strata/core/conversation_tier.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <set>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <utility>
#ifdef _WIN32
#include <stdlib.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
extern char** environ;
#endif

namespace strata::core {
namespace fs = std::filesystem;
namespace {

constexpr char kMetaMagic[8] = {'S', 'T', 'R', 'T', 'I', 'E', 'R', '\x01'};
constexpr uint32_t kMetaVersion = 1;
constexpr uint64_t kMetaMaxBytes = 64ull << 20;   // a meta file holds ids and marks: far below this

// ---- digest of a parked image (verify)
void hash_ck(SessionHasher& h, const ConversationCheckpoint& c) {
    auto vec = [&](const auto& v) {
        const uint64_t n = v.size();
        h.update(&n, 8);
        if (n) h.update(v.data(), n * sizeof(v[0]));
    };
    vec(c.ids);
    const uint64_t ni = c.imgs.size();
    h.update(&ni, 8);
    for (const auto& k : c.imgs) { h.update(&k.start, 8); h.update(&k.hash, 8); }
    vec(c.gdn); vec(c.ple); vec(c.tails); vec(c.dead); vec(c.block_pos);
    const uint64_t marks[3] = {c.used, c.pinned ? 1u : 0u, c.slot_turn ? 1u : 0u};
    h.update(marks, sizeof marks);
    const uint64_t np = c.stage_parts.size();
    h.update(&np, 8);
    for (const auto& p : c.stage_parts) hash_ck(h, p);
}
void hash_buf(SessionHasher& h, const ConversationBuffer& b) {
    const uint64_t n = b.size();
    h.update(&n, 8);
    b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t) { h.update(p, c); return true; });
}
void hash_image(SessionHasher& h, const SavedConversation& s) {
    h.update(s.geometry.data(), sizeof(s.geometry));
    const int64_t head[5] = {s.layer_lo, s.layer_hi, s.cvec ? 1 : 0, s.draftless ? 1 : 0, (int64_t) s.checkpoints.size()};
    h.update(head, sizeof head);
    hash_ck(h, s.live);
    for (const auto& c : s.checkpoints) hash_ck(h, c);
    const uint64_t nk = s.kv.size();
    h.update(&nk, 8);
    for (const auto& k : s.kv) {
        const int64_t m[7] = {k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim};
        h.update(m, sizeof m);
        hash_buf(h, k.k); hash_buf(h, k.v); hash_buf(h, k.k_scale); hash_buf(h, k.v_scale); hash_buf(h, k.pooled);
    }
    const uint64_t ns = s.stage_images.size();
    h.update(&ns, 8);
    for (const auto& p : s.stage_images) hash_image(h, p);
}

// one part of an image against another, its stage images aside (a part file holds none)
bool same_part(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        a.draftless != b.draftless || !conversation_checkpoint_same(a.live, b.live) ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!conversation_checkpoint_same(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!conversation_kv_same(a.kv[i], b.kv[i])) return false;
    return true;
}

// `n` tokens at `ids` with the pictures `imgs` start `full` (whose pictures are `full_imgs`): ConversationCache's
// prefix rule (drop_superseded_by_prefix's `starts`)
template<class Token>
bool starts(const int32_t* ids, size_t n, const std::vector<ConversationImageKey>& imgs, const std::vector<Token>& full,
            const std::vector<ConversationImageKey>& full_imgs) {
    if (n == 0 || n > full.size() || !std::equal(ids, ids + n, full.begin())) return false;
    size_t j = 0;
    for (const auto& image : full_imgs) {
        if (image.start >= (int64_t) n) continue;
        if (j == imgs.size() || !(imgs[j++] == image)) return false;
    }
    return j == imgs.size();
}

std::string hex16(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}
bool parse_hex16(const std::string& s, uint64_t& v) {
    if (s.size() != 16) return false;
    v = 0;
    for (char c : s) {
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return false;
        v = (v << 4) | (uint64_t) d;
    }
    return true;
}

// ---- the meta file: the commit record
struct MetaOut {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); }
    void u64(uint64_t v) { raw(&v, 8); }
    void u8(uint8_t v) { b.push_back(v); }
    void ids(const std::vector<int32_t>& v) { u64(v.size()); raw(v.data(), v.size() * 4); }
    void imgs(const std::vector<ConversationImageKey>& v) {
        u64(v.size());
        for (const auto& k : v) { u64((uint64_t) k.start); u64(k.hash); }
    }
};
struct MetaIn {
    const uint8_t* p;
    size_t left;
    bool raw(void* d, size_t n) { if (n > left) return false; std::memcpy(d, p, n); p += n; left -= n; return true; }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool u8(uint8_t& v) { return raw(&v, 1); }
    bool ids(std::vector<int32_t>& v) {
        uint64_t n = 0;
        if (!u64(n) || n > left / 4) return false;
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * 4);
    }
    bool imgs(std::vector<ConversationImageKey>& v) {
        uint64_t n = 0;
        if (!u64(n) || n > left / 16) return false;
        v.resize((size_t) n);
        for (auto& k : v) {
            uint64_t s = 0;
            if (!u64(s) || !u64(k.hash)) return false;
            k.start = (int64_t) s;
        }
        return true;
    }
};

// a small file written whole: temporary name, flush, rename, folder flush (POSIX); false + `why`
bool write_small(const std::string& dir, const std::string& leaf, const std::vector<uint8_t>& bytes, bool durable,
                 std::string& why) {
    static std::atomic<uint64_t> seq{0};
    const std::string tmp = dir + "/." + leaf + "." + hex16(++seq ^ (uint64_t) std::chrono::steady_clock::now()
                                                                        .time_since_epoch().count()) + ".tmp";
    const std::string dst = dir + "/" + leaf;
#ifdef _WIN32
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { why = "cannot create " + tmp; return false; }
        f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
        f.flush();
        if (!f) { std::error_code ec; fs::remove(tmp, ec); why = "cannot write " + tmp; return false; }
    }
    (void) durable;
    std::error_code ec;
    fs::rename(tmp, dst, ec);
    if (ec) { fs::remove(tmp, ec); why = "cannot rename " + tmp + ": " + ec.message(); return false; }
    return true;
#else
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { why = "cannot create " + tmp + ": " + std::strerror(errno); return false; }
    const uint8_t* p = bytes.data();
    size_t n = bytes.size();
    bool ok = true;
    while (n && ok) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { why = std::string("write: ") + (w < 0 ? std::strerror(errno) : "short write"); ok = false; break; }
        p += w; n -= (size_t) w;
    }
    if (ok && durable && ::fsync(fd) != 0) { why = std::string("flush: ") + std::strerror(errno); ok = false; }
    if (::close(fd) != 0 && ok) { why = std::string("close: ") + std::strerror(errno); ok = false; }
    if (ok && ::rename(tmp.c_str(), dst.c_str()) != 0) { why = std::string("rename: ") + std::strerror(errno); ok = false; }
    if (!ok) { ::unlink(tmp.c_str()); return false; }
    if (durable) {
        const int d = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (d >= 0) {
            if (::fsync(d) != 0 && errno != EINVAL) { why = std::string("folder flush: ") + std::strerror(errno); ::close(d); return false; }
            ::close(d);
        }
    }
    return true;
#endif
}

bool read_small(const std::string& path, std::vector<uint8_t>& out, uint64_t max) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n < 0 || (uint64_t) n > max) return false;
    f.seekg(0);
    out.resize((size_t) n);
    f.read(reinterpret_cast<char*>(out.data()), n);
    return (bool) f;
}

int64_t mtime_s(const std::string& path, int64_t fallback) {
#ifdef _WIN32
    (void) path;
    return fallback;
#else
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? (int64_t) st.st_mtime : fallback;
#endif
}
void touch(const std::string& path) {
#ifndef _WIN32
    (void) ::utimensat(AT_FDCWD, path.c_str(), nullptr, 0);
#else
    (void) path;
#endif
}

uint64_t file_size(const std::string& path) {
    std::error_code ec;
    const auto n = fs::file_size(path, ec);
    return ec ? UINT64_MAX : (uint64_t) n;
}

} // namespace

uint64_t conversation_image_digest(const SavedConversation& image) {
    SessionHasher h(0x5354524954494552ull);
    hash_image(h, image);
    return h.digest();
}

bool conversation_tier_fingerprint(uint64_t config, const std::string& exe, const std::vector<std::string>& args,
                                   uint64_t& out, std::string& error) {
    uint64_t binary = 0;
    if (!session_model_fingerprint({{"engine binary", exe}}, binary, error)) return false;
    std::vector<std::string> env;
#ifdef _WIN32
    char** e = _environ;
#else
    char** e = environ;
#endif
    for (; e != nullptr && *e != nullptr; ++e) {
        const std::string kv = *e;
        if (kv.rfind("STRATA_", 0) == 0 && kv.rfind("STRATA_CONV_TIER_", 0) != 0) env.push_back(kv);
    }
    std::sort(env.begin(), env.end());
    SessionIdentityBuilder b(0x5354524954494552ull);
    b.u64("config", config);
    b.u64("binary", binary);
    b.i64("args", (int64_t) args.size());
    for (const auto& a : args) b.str("arg", a);
    b.i64("env", (int64_t) env.size());
    for (const auto& v : env) b.str("var", v);
    out = b.digest();
    return true;
}

ConversationTier::ConversationTier(ConversationTierConfig config) : cfg_(std::move(config)) {
    while (cfg_.dir.size() > 1 && cfg_.dir.back() == '/') cfg_.dir.pop_back();
}

ConversationTier::~ConversationTier() {
    std::vector<std::unique_ptr<SavedConversation>> drop;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        for (size_t i = 0; i < entries_.size();) {
            if (entries_[i]->state == State::queued) {
                drop.push_back(std::move(entries_[i]->image));
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                continue;
            }
            ++i;
        }
    }
    work_.notify_all();
    if (writer_.joinable()) writer_.join();
}

int64_t ConversationTier::now() const {
    return cfg_.clock ? cfg_.clock() : (int64_t) std::time(nullptr);
}

int64_t ConversationTier::day_of(int64_t unix_s) {
    const std::time_t t = (std::time_t) unix_s;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return (int64_t) (tm.tm_year + 1900) * 10000 + (tm.tm_mon + 1) * 100 + tm.tm_mday;
}

std::string ConversationTier::path(uint64_t key, int part) const {
    return cfg_.dir + "/" + hex16(key) + (part < 0 ? std::string(".meta") : "." + std::to_string(part) + ".ses");
}

void ConversationTier::log_event(const std::string& what) {
    if (cfg_.log) cfg_.log(what + "; " + status_locked());
}

std::string ConversationTier::status_locked() const {
    const ConversationTierStats& s = st_;
    char b[1024];
    std::snprintf(b, sizeof b,
                  "files=%llu parts=%llu disk_bytes=%llu written_today=%llu budget=%llu budget_left=%llu spills=%llu "
                  "written_bytes=%llu restores=%llu restores_queue=%llu read_bytes=%llu misses=%llu drops_small=%llu "
                  "drops_budget=%llu drops_busy=%llu drops_ram=%llu drops_cap=%llu drops_io=%llu kept=%llu "
                  "superseded=%llu expired=%llu evicted=%llu pending=%llu pending_bytes=%llu foreign=%llu verified=%llu",
                  (unsigned long long) s.files, (unsigned long long) s.parts, (unsigned long long) s.disk_bytes,
                  (unsigned long long) s.written_today, (unsigned long long) cfg_.budget_bytes,
                  (unsigned long long) (cfg_.budget_bytes > s.written_today ? cfg_.budget_bytes - s.written_today : 0),
                  (unsigned long long) s.spills, (unsigned long long) s.written_bytes, (unsigned long long) s.restores,
                  (unsigned long long) s.restores_queue, (unsigned long long) s.read_bytes,
                  (unsigned long long) s.misses, (unsigned long long) s.drops_small,
                  (unsigned long long) s.drops_budget, (unsigned long long) s.drops_busy,
                  (unsigned long long) s.drops_ram, (unsigned long long) s.drops_cap, (unsigned long long) s.drops_io,
                  (unsigned long long) s.kept, (unsigned long long) s.superseded, (unsigned long long) s.expired,
                  (unsigned long long) s.evicted, (unsigned long long) s.pending, (unsigned long long) s.pending_bytes,
                  (unsigned long long) s.foreign, (unsigned long long) s.verified);
    return b;
}

ConversationTierStats ConversationTier::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    ConversationTierStats s = st_;
    s.budget = cfg_.budget_bytes;
    s.budget_left = cfg_.budget_bytes > s.written_today ? cfg_.budget_bytes - s.written_today : 0;
    return s;
}

std::string ConversationTier::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_locked();
}

void ConversationTier::roll_day_locked() {
    const int64_t d = day_of(now());
    if (d != day_) {
        day_ = d;
        st_.written_today = 0;
    }
}

void ConversationTier::save_budget_locked() {
    const std::string text = std::to_string(day_) + " " + std::to_string(st_.written_today) + "\n";
    std::string why;
    (void) write_small(cfg_.dir, "budget", std::vector<uint8_t>(text.begin(), text.end()), false, why);
}

size_t ConversationTier::find_locked(uint64_t key) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i]->key == key) return i;
    return SIZE_MAX;
}

void ConversationTier::remove_files(uint64_t key, size_t parts) {
    std::error_code ec;
    fs::remove(path(key, -1), ec);   // the commit record first: a set without it is never used
    for (size_t k = 0; k < parts; ++k) fs::remove(path(key, (int) k), ec);
}

void ConversationTier::remove_entry_locked(size_t i) {
    const EntryPtr e = entries_[i];
    if (e->state == State::disk) {
        remove_files(e->key, e->parts.size());
        st_.files -= 1;
        st_.parts -= e->parts.size();
        st_.disk_bytes -= std::min(st_.disk_bytes, e->bytes);
    } else if (e->state == State::queued) {
        st_.pending -= 1;
        st_.pending_bytes -= std::min(st_.pending_bytes, e->bytes);
    }
    entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
}

bool ConversationTier::sweep_locked(uint64_t incoming) {
    const int64_t t = now();
    size_t gone = 0;
    for (size_t i = 0; i < entries_.size();) {
        const Entry& e = *entries_[i];
        if (e.state == State::disk && e.readers == 0 && cfg_.ttl_s > 0 && t - e.used > cfg_.ttl_s) {
            remove_entry_locked(i);
            ++st_.expired;
            ++gone;
            continue;
        }
        ++i;
    }
    if (gone) log_event("expired " + std::to_string(gone) + " (unused for " + std::to_string(cfg_.ttl_s / 3600) + " h)");
    size_t evicted = 0;
    bool room = true;
    while (st_.disk_bytes + incoming > cfg_.cap_bytes || st_.files + 1 > cfg_.max_images) {
        size_t victim = SIZE_MAX;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = *entries_[i];
            if (e.state == State::disk && e.readers == 0 && (victim == SIZE_MAX || e.used < entries_[victim]->used))
                victim = i;
        }
        if (victim == SIZE_MAX) { room = false; break; }
        remove_entry_locked(victim);
        ++st_.evicted;
        ++evicted;
    }
    if (evicted) log_event("removed " + std::to_string(evicted) + " least recently used (disk cap)");
    return room;
}

size_t ConversationTier::cover_locked(const SavedConversation& image) const {
    const auto& ids = image.live.ids;
    const auto& imgs = image.live.imgs;
    const size_t L = ids.size();
    for (size_t i = entries_.size(); i-- > 0;) {
        const Entry& e = *entries_[i];
        if (e.cvec != image.cvec) continue;
        // a later state of this conversation is held already
        if (starts(ids.data(), L, imgs, e.ids, e.imgs)) return i;
        // it holds this conversation all but fewer than regrow_tokens
        size_t m = starts(e.ids.data(), e.ids.size(), e.imgs, ids, imgs) ? e.ids.size() : 0;
        for (const auto& c : e.checks) {
            const int32_t* p = c.own.empty() ? e.ids.data() : c.own.data();
            if (c.len > m && starts(p, c.len, c.imgs, ids, imgs)) m = c.len;
        }
        if (m > 0 && (int64_t) (L - m) < cfg_.regrow_tokens) return i;
    }
    return SIZE_MAX;
}

bool ConversationTier::start(std::string& error) {
    error.clear();
    if (cfg_.dir.empty()) { error = "no folder"; return false; }
    std::error_code ec;
    const bool made = fs::create_directories(cfg_.dir, ec);
    if (ec || !fs::is_directory(cfg_.dir, ec)) {
        error = "cannot create the folder " + cfg_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    if (made) (void) ::chmod(cfg_.dir.c_str(), 0700);
#else
    (void) made;
#endif
    {
        std::lock_guard<std::mutex> lk(mu_);
        day_ = day_of(now());
        std::vector<uint8_t> b;
        if (read_small(cfg_.dir + "/budget", b, 256)) {
            long long d = 0;
            unsigned long long n = 0;
            const std::string s(b.begin(), b.end());
            if (std::sscanf(s.c_str(), "%lld %llu", &d, &n) == 2 && d == day_) st_.written_today = n;
        }
        std::random_device rd;
        next_key_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    }
    scan(error);
    if (!error.empty()) return false;
    try {
        writer_ = std::thread([this] { writer_loop(); });
    } catch (const std::exception& e) {
        error = std::string("cannot start the writer: ") + e.what();
        return false;
    }
    started_ = true;
    std::lock_guard<std::mutex> lk(mu_);
    log_event(std::string("on: ") + cfg_.dir + (cfg_.warm ? " (warm)" : ""));
    return true;
}

void ConversationTier::scan(std::string& error) {
    std::error_code ec;
    std::vector<std::string> temps;
    std::unordered_map<uint64_t, std::vector<int>> parts;   // key -> its part numbers
    std::vector<uint64_t> metas;
    for (fs::directory_iterator it(cfg_.dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() > 5 && name[0] == '.' && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            temps.push_back(name);   // a write an earlier run did not finish
            continue;
        }
        uint64_t key = 0;
        if (name.size() < 17 || name[16] != '.' || !parse_hex16(name.substr(0, 16), key)) continue;
        const std::string rest = name.substr(17);
        if (rest == "meta") { metas.push_back(key); continue; }
        if (rest.size() > 4 && rest.compare(rest.size() - 4, 4, ".ses") == 0) {
            const std::string num = rest.substr(0, rest.size() - 4);
            if (!num.empty() && num.size() < 4 && std::all_of(num.begin(), num.end(), [](char c) { return c >= '0' && c <= '9'; }))
                parts[key].push_back(std::stoi(num));
        }
    }
    if (ec) { error = "cannot list " + cfg_.dir + ": " + ec.message(); return; }
    for (const auto& t : temps) fs::remove(cfg_.dir + "/" + t, ec);
    std::lock_guard<std::mutex> lk(mu_);
    std::set<uint64_t> keep;
    size_t removed = 0, indexed = 0;
    for (uint64_t key : metas) {
        bool ok = false, foreign = false;
        auto e = std::make_shared<Entry>();
        e->key = key;
        if (cfg_.warm) {
            std::vector<uint8_t> b;
            if (read_small(path(key, -1), b, kMetaMaxBytes) && b.size() >= 8 + 8) {
                const uint64_t want = session_hash64(b.data(), b.size() - 8, 0);
                uint64_t have = 0;
                std::memcpy(&have, b.data() + b.size() - 8, 8);
                MetaIn in{b.data(), b.size() - 8};
                char magic[8];
                uint64_t version = 0, model = 0, config = 0, k = 0, created = 0, cvec = 0, nchecks = 0, nparts = 0;
                if (want == have && in.raw(magic, 8) && std::memcmp(magic, kMetaMagic, 8) == 0 && in.u64(version) &&
                    version == kMetaVersion && in.u64(model) && in.u64(config) && in.u64(k) && k == key) {
                    if (model != cfg_.id.model || config != cfg_.id.config) {
                        foreign = true;
                    } else if (in.u64(created) && in.u64(e->digest) && in.u64(cvec) && in.ids(e->ids) &&
                               in.imgs(e->imgs) && in.u64(nchecks) && nchecks <= in.left / 8) {
                        e->created = (int64_t) created;
                        e->cvec = cvec != 0;
                        bool good = true;
                        e->checks.resize((size_t) nchecks);
                        for (auto& c : e->checks) {
                            uint64_t len = 0;
                            uint8_t prefix = 0;
                            good = good && in.u64(len) && in.u8(prefix);
                            if (good && !prefix) good = in.ids(c.own) && c.own.size() == len;
                            good = good && in.imgs(c.imgs) && (!prefix || len <= e->ids.size());
                            c.len = (size_t) len;
                            if (!good) break;
                        }
                        good = good && in.u64(nparts) && nparts > 0 && nparts < 64;
                        if (good) {
                            e->parts.resize((size_t) nparts);
                            e->part_bytes.resize((size_t) nparts);
                            for (size_t p = 0; p < nparts && good; ++p) {
                                auto& m = e->parts[p];
                                uint64_t bytes = 0, nck = 0;
                                uint8_t dl = 0, lp = 0, ls = 0;
                                good = in.u64(bytes) && in.u8(dl) && in.u8(lp) && in.u8(ls) && in.u64(nck) &&
                                       nck <= in.left / 2;
                                if (!good) break;
                                m.draftless = dl; m.live_pinned = lp; m.live_slot_turn = ls;
                                m.pinned.resize((size_t) nck);
                                m.slot_turn.resize((size_t) nck);
                                for (size_t c = 0; c < nck && good; ++c) good = in.u8(m.pinned[c]) && in.u8(m.slot_turn[c]);
                                e->part_bytes[p] = bytes;
                                good = good && file_size(path(key, (int) p)) == bytes;
                            }
                        }
                        ok = good && in.left == 0;
                    }
                }
            }
        }
        if (foreign) {   // another engine's (another build, model or setting): kept until its time to live is over
            const int64_t used = mtime_s(path(key, -1), now());
            if (cfg_.ttl_s > 0 && now() - used > cfg_.ttl_s) {
                remove_files(key, parts.count(key) ? (size_t) *std::max_element(parts[key].begin(), parts[key].end()) + 1 : 0);
                ++removed;
            } else {
                keep.insert(key);
                ++st_.foreign;
            }
            continue;
        }
        if (!ok) {
            fs::remove(path(key, -1), ec);
            ++removed;
            continue;
        }
        e->state = State::disk;
        e->used = mtime_s(path(key, -1), e->created);
        e->bytes = file_size(path(key, -1));
        for (uint64_t b : e->part_bytes) e->bytes += b;
        keep.insert(key);
        st_.files += 1;
        st_.parts += e->parts.size();
        st_.disk_bytes += e->bytes;
        entries_.push_back(e);
        ++indexed;
    }
    // part files without a commit record (a run ended between the parts and the meta file, or not warm)
    for (auto& [key, nums] : parts) {
        if (keep.count(key)) continue;
        for (int n : nums) fs::remove(path(key, n), ec);
        ++removed;
    }
    std::sort(entries_.begin(), entries_.end(), [](const EntryPtr& a, const EntryPtr& b) { return a->used < b->used; });
    if (removed || indexed || !temps.empty() || st_.foreign)
        log_event("start: indexed " + std::to_string(indexed) + ", removed " + std::to_string(removed) + " set" +
                  (removed == 1 ? "" : "s") + " and " + std::to_string(temps.size()) + " unfinished file" +
                  (temps.size() == 1 ? "" : "s") + ", " + std::to_string(st_.foreign) + " of another engine kept");
}

void ConversationTier::evicted(SavedConversation&& image, bool spill) {
    if (!started_) return;
    const size_t tokens = image.live.ids.size();
    if (!spill) {
        if ((int64_t) tokens >= cfg_.min_tokens) {
            std::lock_guard<std::mutex> lk(mu_);
            ++st_.drops_ram;
        }
        return;   // the caller frees it now
    }
    try {
        std::lock_guard<std::mutex> lk(mu_);
        if ((int64_t) tokens < cfg_.min_tokens) {
            ++st_.drops_small;
            log_event("dropped " + std::to_string(tokens) + " tokens (below the " + std::to_string(cfg_.min_tokens) +
                      " minimum)");
            return;
        }
        const size_t cover = cover_locked(image);
        if (cover != SIZE_MAX) {
            Entry& e = *entries_[cover];
            e.used = now();
            if (e.state == State::disk) touch(path(e.key, -1));
            ++st_.kept;
            log_event("kept " + std::to_string(tokens) + " tokens: the file of " + std::to_string(e.ids.size()) +
                      " tokens holds them (nothing written)");
            return;
        }
        const uint64_t est = image.bytes();
        roll_day_locked();
        if (cfg_.budget_bytes == 0 || st_.written_today + st_.pending_bytes + est > cfg_.budget_bytes) {
            ++st_.drops_budget;
            log_event("dropped " + std::to_string(tokens) + " tokens (" + std::to_string(est >> 20) +
                      " MiB: the day's write budget is used up)");
            return;
        }
        if (st_.pending > 0 && st_.pending_bytes + est > cfg_.inflight_bytes) {
            ++st_.drops_busy;
            log_event("dropped " + std::to_string(tokens) + " tokens (the writer is behind: " +
                      std::to_string(st_.pending_bytes >> 20) + " MiB waiting)");
            return;
        }
        auto e = std::make_shared<Entry>();
        do e->key = ++next_key_ * 0x9E3779B97F4A7C15ull;
        while (e->key == 0 || find_locked(e->key) != SIZE_MAX || fs::exists(path(e->key, -1)));
        e->ids = image.live.ids;
        e->imgs = image.live.imgs;
        e->cvec = image.cvec;
        for (const auto& c : image.checkpoints) {
            Entry::Check k;
            k.len = c.ids.size();
            k.imgs = c.imgs;
            if (!(k.len <= e->ids.size() && std::equal(c.ids.begin(), c.ids.end(), e->ids.begin()))) k.own = c.ids;
            e->checks.push_back(std::move(k));
        }
        auto marks = [](const SavedConversation& p) {
            Entry::Marks m;
            m.draftless = p.draftless;
            m.live_pinned = p.live.pinned;
            m.live_slot_turn = p.live.slot_turn;
            for (const auto& c : p.checkpoints) { m.pinned.push_back(c.pinned); m.slot_turn.push_back(c.slot_turn); }
            return m;
        };
        e->parts.push_back(marks(image));
        for (const auto& s : image.stage_images) e->parts.push_back(marks(s));
        e->bytes = est;
        e->created = e->used = now();
        e->image = std::make_unique<SavedConversation>(std::move(image));
        entries_.push_back(e);
        ++st_.pending;
        st_.pending_bytes += est;
        work_.notify_one();
    } catch (const std::exception&) {   // (bad_alloc, a filesystem error): dropped as without the tier
        std::lock_guard<std::mutex> lk(mu_);
        ++st_.drops_io;
    }
}

template<class Token>
ConversationTier::Match ConversationTier::best(const std::vector<Token>& prompt,
                                               const std::vector<ConversationImageKey>& images, bool cvec) const {
    Match best;
    if (!started_) return best;
    std::lock_guard<std::mutex> lk(mu_);
    // the prompt's last token always starts the next window: a match is shorter than the prompt (conversation_prefix)
    auto prefix = [&](const int32_t* ids, size_t n, const std::vector<ConversationImageKey>& imgs) -> int64_t {
        if (n >= prompt.size()) return 0;
        return starts(ids, n, imgs, prompt, images) ? (int64_t) n : 0;
    };
    for (size_t i = entries_.size(); i-- > 0;) {
        const Entry& e = *entries_[i];
        if (e.cvec != cvec || e.cancelled) continue;
        auto take = [&](int64_t n, bool live) {
            if (n > best.tokens) best = {e.key, n, live, e.parts.empty() ? false : e.parts[0].draftless, e.parts.size()};
        };
        take(prefix(e.ids.data(), e.ids.size(), e.imgs), true);
        for (const auto& c : e.checks) take(prefix(c.own.empty() ? e.ids.data() : c.own.data(), c.len, c.imgs), false);
    }
    return best;
}

template ConversationTier::Match ConversationTier::best(const std::vector<int32_t>&,
                                                        const std::vector<ConversationImageKey>&, bool) const;
template ConversationTier::Match ConversationTier::best(const std::vector<int64_t>&,
                                                        const std::vector<ConversationImageKey>&, bool) const;

bool ConversationTier::load(uint64_t key, const std::vector<SessionReadLimits>& limits, SavedConversation& out,
                            std::string& why, LoadInfo* info) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!started_) { why = "off"; return false; }
    EntryPtr e;
    {
        std::unique_lock<std::mutex> lk(mu_);
        size_t i = find_locked(key);
        if (i != SIZE_MAX && entries_[i]->state == State::writing) {
            const EntryPtr w = entries_[i];
            changed_.wait(lk, [&] { return w->state != State::writing || w->cancelled || find_locked(key) == SIZE_MAX; });
            i = find_locked(key);
        }
        if (i == SIZE_MAX || entries_[i]->cancelled || entries_[i]->state == State::writing) {
            ++st_.misses;
            why = "gone (removed meanwhile)";
            log_event("miss: " + why);
            return false;
        }
        e = entries_[i];
        if (e->state == State::queued) {   // never written: the image itself comes back
            std::unique_ptr<SavedConversation> img = std::move(e->image);
            const size_t tokens = e->ids.size();
            remove_entry_locked(i);
            ++st_.restores;
            ++st_.restores_queue;
            if (info) { info->from_queue = true; info->bytes = 0; }
            out = std::move(*img);
            if (info) info->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            log_event("restored " + std::to_string(tokens) + " tokens from the write queue");
            return true;
        }
        ++e->readers;
    }
    const size_t n = e->parts.size();
    std::vector<SavedConversation> part(n);
    std::vector<std::string> err(n);
    std::vector<SessionStatus> status(n);
    std::vector<size_t> got(n, 0);
    std::vector<char> ok(n, 0);
    bool good = limits.size() >= n;
    if (!good) why = "read limits for " + std::to_string(limits.size()) + " of " + std::to_string(n) + " parts";
    if (good && cfg_.fault) {
        if (const int f = cfg_.fault("read")) { good = false; why = std::string("read: ") + std::strerror(f); }
    }
    if (good) {
        auto job = [&](size_t k, bool spawned) {
            try {
                if (cfg_.thread_start && spawned) cfg_.thread_start();   // (never the caller's own thread)
                ok[k] = session_file_read(path(key, (int) k), cfg_.id, part[k], got[k], err[k], limits[k], &status[k]);
            } catch (const std::exception& x) {
                err[k] = x.what();
                status[k].error = SessionError::memory;
            }
        };
        std::vector<std::thread> threads;
        try {
            for (size_t k = 1; k < n; ++k) threads.emplace_back(job, k, true);
        } catch (const std::exception&) {}
        job(0, false);
        for (size_t k = threads.size() + 1; k < n; ++k) job(k, false);   // a thread that could not start: here
        for (auto& t : threads) t.join();
        for (size_t k = 0; k < n && good; ++k)
            if (!ok[k]) { good = false; why = err[k]; }
    }
    bool keep_files = false;
    for (size_t k = 0; k < n; ++k) keep_files = keep_files || (!ok[k] && status[k].error == SessionError::memory);
    // the parts against the commit record: the same conversation, the marks' checkpoint counts
    if (good) {
        for (size_t k = 0; k < n && good; ++k) {
            const auto& p = part[k];
            const auto& m = e->parts[k];
            if (p.live.ids != e->ids || p.live.imgs != e->imgs || p.cvec != e->cvec ||
                p.checkpoints.size() != m.pinned.size() || (k == 0 && p.checkpoints.size() != e->checks.size())) {
                good = false;
                why = "part " + std::to_string(k) + " does not match its commit record";
            }
        }
    }
    uint64_t bytes = 0;
    if (good) {
        for (size_t k = 0; k < n; ++k) {
            auto& p = part[k];
            const auto& m = e->parts[k];
            p.draftless = m.draftless;
            p.live.pinned = m.live_pinned;
            p.live.slot_turn = m.live_slot_turn;
            for (size_t c = 0; c < p.checkpoints.size(); ++c) {
                p.checkpoints[c].pinned = m.pinned[c] != 0;
                p.checkpoints[c].slot_turn = m.slot_turn[c] != 0;
            }
            bytes += got[k];
        }
        SavedConversation img = std::move(part[0]);
        try {
            for (size_t k = 1; k < n; ++k) img.stage_images.push_back(std::move(part[k]));
        } catch (const std::bad_alloc&) {
            good = false;
            keep_files = true;
            why = "out of memory";
        }
        if (good && cfg_.verify && e->digest != 0 && conversation_image_digest(img) != e->digest) {
            good = false;
            why = "verify: the image read back differs from the one evicted (digest)";
        }
        if (good) out = std::move(img);
    }
    std::lock_guard<std::mutex> lk(mu_);
    --e->readers;
    const size_t i = find_locked(key);
    if (!good) {
        ++st_.misses;
        if (!keep_files && i != SIZE_MAX && e->readers == 0) remove_entry_locked(i);
        log_event("miss: " + why + (keep_files ? " (the files stay)" : " (the files go)"));
        return false;
    }
    if (cfg_.verify && e->digest != 0) ++st_.verified;
    e->used = now();
    touch(path(key, -1));
    ++st_.restores;
    st_.read_bytes += bytes;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (info) { info->from_queue = false; info->bytes = bytes; info->ms = ms; }
    char b[160];
    std::snprintf(b, sizeof b, "restored %zu tokens from disk in %.1f ms (read %.1f MB, %zu files%s)", e->ids.size(), ms,
                  bytes / 1e6, n, cfg_.verify && e->digest != 0 ? ", digest verified" : "");
    log_event(b);
    return true;
}

void ConversationTier::reject(uint64_t key, const std::string& why) {
    std::lock_guard<std::mutex> lk(mu_);
    ++st_.misses;
    const size_t i = find_locked(key);
    if (i != SIZE_MAX && entries_[i]->state == State::disk && entries_[i]->readers == 0) remove_entry_locked(i);
    log_event("miss: refused (" + why + "); the files go");
}

void ConversationTier::clear() {
    if (!started_) return;
    std::vector<std::unique_ptr<SavedConversation>> drop;
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (size_t i = 0; i < entries_.size();) {
        Entry& e = *entries_[i];
        if (e.state == State::writing) { e.cancelled = true; ++i; ++n; continue; }
        if (e.state == State::queued) drop.push_back(std::move(e.image));
        remove_entry_locked(i);
        ++n;
    }
    log_event("cleared " + std::to_string(n) + " (non-finite logits)");
}

void ConversationTier::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    idle_.wait(lk, [&] {
        if (busy_) return false;
        for (const auto& e : entries_) if (e->state == State::queued) return false;
        return true;
    });
}

bool ConversationTier::write_set(Entry& e, uint64_t& total, std::string& why) {
    SavedConversation& image = *e.image;
    const size_t n = 1 + image.stage_images.size();
    std::vector<std::string> err(n);
    std::vector<size_t> bytes(n, 0);
    std::vector<char> ok(n, 0);
    auto src = [&](size_t k) -> const SavedConversation& { return k == 0 ? image : image.stage_images[k - 1]; };
    auto job = [&](size_t k, bool spawned) {
        try {
            if (cfg_.thread_start && spawned) cfg_.thread_start();
            SessionWriteOptions o;
            o.min_free_bytes = cfg_.min_free_bytes;
            o.durable = cfg_.durable;
            o.fault = cfg_.fault;
            ok[k] = session_file_write(path(e.key, (int) k), src(k), cfg_.id, bytes[k], err[k], o);
        } catch (const std::exception& x) {
            err[k] = x.what();
        }
    };
    {
        std::vector<std::thread> threads;
        try {
            for (size_t k = 1; k < n; ++k) threads.emplace_back(job, k, true);
        } catch (const std::exception&) {}
        job(0, false);
        for (size_t k = threads.size() + 1; k < n; ++k) job(k, false);
        for (auto& t : threads) t.join();
    }
    for (size_t k = 0; k < n; ++k)
        if (!ok[k]) { why = err[k]; remove_files(e.key, n); return false; }
    e.part_bytes.assign(bytes.begin(), bytes.end());
    // the commit record, after every part is in place
    MetaOut m;
    m.raw(kMetaMagic, 8);
    m.u64(kMetaVersion);
    m.u64(cfg_.id.model);
    m.u64(cfg_.id.config);
    m.u64(e.key);
    m.u64((uint64_t) e.created);
    m.u64(e.digest);
    m.u64(e.cvec ? 1 : 0);
    m.ids(e.ids);
    m.imgs(e.imgs);
    m.u64(e.checks.size());
    for (const auto& c : e.checks) {
        m.u64(c.len);
        m.u8(c.own.empty() ? 1 : 0);
        if (!c.own.empty()) m.ids(c.own);
        m.imgs(c.imgs);
    }
    m.u64(n);
    for (size_t k = 0; k < n; ++k) {
        const auto& p = e.parts[k];
        m.u64(bytes[k]);
        m.u8(p.draftless); m.u8(p.live_pinned); m.u8(p.live_slot_turn);
        m.u64(p.pinned.size());
        for (size_t c = 0; c < p.pinned.size(); ++c) { m.u8(p.pinned[c]); m.u8(p.slot_turn[c]); }
    }
    m.u64(session_hash64(m.b.data(), m.b.size(), 0));
    if (cfg_.fault) {
        if (const int f = cfg_.fault("meta")) { why = std::string("meta: ") + std::strerror(f); remove_files(e.key, n); return false; }
    }
    if (!write_small(cfg_.dir, hex16(e.key) + ".meta", m.b, cfg_.durable, why)) {
        why = "meta: " + why;
        remove_files(e.key, n);
        return false;
    }
    total = m.b.size();
    for (size_t b : bytes) total += b;
    if (cfg_.verify) {   // every part read back and compared with what was evicted
        for (size_t k = 0; k < n; ++k) {
            SavedConversation back;
            size_t got = 0;
            std::string e2;
            if (!session_file_read(path(e.key, (int) k), cfg_.id, back, got, e2)) {
                why = "verify: reading part " + std::to_string(k) + " back: " + e2;
                remove_files(e.key, n);
                return false;
            }
            const auto& p = e.parts[k];
            back.draftless = p.draftless;
            back.live.pinned = p.live_pinned;
            back.live.slot_turn = p.live_slot_turn;
            for (size_t c = 0; c < back.checkpoints.size() && c < p.pinned.size(); ++c) {
                back.checkpoints[c].pinned = p.pinned[c] != 0;
                back.checkpoints[c].slot_turn = p.slot_turn[c] != 0;
            }
            if (!same_part(back, src(k))) {
                why = "verify: part " + std::to_string(k) + " read back differs from the evicted image";
                remove_files(e.key, n);
                return false;
            }
        }
    }
    return true;
}

void ConversationTier::writer_loop() {
    if (cfg_.thread_start) cfg_.thread_start();
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        EntryPtr e;
        for (const auto& x : entries_)
            if (x->state == State::queued) { e = x; break; }
        if (!e) {
            idle_.notify_all();
            if (stop_) return;
            // an idle minute: the day's budget and the time to live move on without writes
            if (!work_.wait_for(lk, std::chrono::seconds(60), [&] {
                    if (stop_) return true;
                    for (const auto& x : entries_) if (x->state == State::queued) return true;
                    return false;
                })) {
                const int64_t d = day_;
                roll_day_locked();
                if (d != day_) { save_budget_locked(); log_event("a new day: the write budget starts again"); }
                (void) sweep_locked(0);
            }
            continue;
        }
        if (stop_) return;
        const auto t0 = std::chrono::steady_clock::now();
        roll_day_locked();
        e->state = State::writing;
        busy_ = true;
        --st_.pending;
        st_.pending_bytes -= std::min(st_.pending_bytes, e->bytes);
        const bool room = sweep_locked(e->bytes);
        bool ok = false;
        uint64_t total = 0;
        std::string why;
        if (!room) {
            why = "cap";
        } else {
            lk.unlock();
            try {
                if (cfg_.verify) e->digest = conversation_image_digest(*e->image);
                ok = write_set(*e, total, why);
            } catch (const std::exception& x) {
                why = x.what();
                ok = false;
            }
            e->image.reset();   // the RAM goes back now
            lk.lock();
        }
        e->image.reset();
        const size_t i = find_locked(e->key);
        busy_ = false;
        if (e->cancelled || i == SIZE_MAX) {
            if (ok) remove_files(e->key, e->parts.size());
            if (i != SIZE_MAX) entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
        } else if (ok) {
            e->state = State::disk;
            e->bytes = total;
            e->used = now();
            st_.files += 1;
            st_.parts += e->parts.size();
            st_.disk_bytes += total;
            st_.written_today += total;
            st_.written_bytes += total;
            ++st_.spills;
            if (cfg_.verify) ++st_.verified;
            // the older files of this conversation (its tokens start this one's) go
            size_t gone = 0;
            for (size_t j = 0; j < entries_.size();) {
                const Entry& d = *entries_[j];
                if (entries_[j] != e && d.state == State::disk && d.readers == 0 && d.cvec == e->cvec &&
                    d.ids.size() < e->ids.size() && starts(d.ids.data(), d.ids.size(), d.imgs, e->ids, e->imgs)) {
                    remove_entry_locked(j);
                    ++st_.superseded;
                    ++gone;
                    continue;
                }
                ++j;
            }
            save_budget_locked();
            char b[200];
            std::snprintf(b, sizeof b, "spilled %zu tokens (%.1f MB, %zu files%s) in %.1f ms%s", e->ids.size(),
                          total / 1e6, e->parts.size() + 1, cfg_.verify ? ", verified" : "",
                          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                          gone ? (", " + std::to_string(gone) + " older file set" + (gone == 1 ? "" : "s") + " of it removed").c_str() : "");
            log_event(b);
        } else {
            if (why == "cap") ++st_.drops_cap; else ++st_.drops_io;
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            log_event("dropped " + std::to_string(e->ids.size()) + " tokens (" +
                      (why == "cap" ? std::string("the disk cap: every file is in use") : "write failed: " + why) + ")");
        }
        changed_.notify_all();
    }
}

} // namespace strata::core
