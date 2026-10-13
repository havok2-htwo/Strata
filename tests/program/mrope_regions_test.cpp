// The M-RoPE table's slot regions (include/strata/program/mrope_regions.hpp): the layout, the bases, when an admission
// copies the main region into a slot's, and a host model of the table as the rope kernels read it (mrope_pos: row *
// 3 + pair % 3) - text slots keep their own positions while an image is read and decoded beside them.  Header-only,
// CPU: no model, no GPU.
#include "strata/program/mrope_regions.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace mr = strata::program::mrope_regions;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// what a rope kernel rotates pair `pair` of a row by (mrope.hpp's mrope_pos, the table set)
int32_t rot(const std::vector<int32_t>& tab, int64_t row, int pair) { return tab[(size_t) row * 3 + pair % 3]; }

// the engine's image layout of one request (generate.cpp, the GENI parse): text at p, an image's cells at (p, p+y,
// p+x), the text after it at p + max(nx, ny), every cell past the prompt continuing from there.  `img` holds the
// first cell, nx and ny of each image.
struct Img { int64_t at, nx, ny; };
std::vector<int32_t> layout(int64_t n, const std::vector<Img>& imgs, int64_t cells) {
    std::vector<int32_t> v((size_t) cells * 3);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        v[(size_t) c * 3] = (int32_t) t; v[(size_t) c * 3 + 1] = (int32_t) h; v[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (k < imgs.size() && imgs[k].at == i) {
            const Img& im = imgs[k++];
            for (int64_t j = 0; j < im.nx * im.ny; ++j) put(i + j, p, p + j / im.nx, p + j % im.nx);
            i += im.nx * im.ny;
            p += im.nx > im.ny ? im.nx : im.ny;
            continue;
        }
        put(i, p, p, p);
        ++p;
        ++i;
    }
    for (int64_t c = n; c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
    return v;
}

// A host model of one device's table and the engine's bookkeeping: region 0 rebuilt at a request's parse
// (upload_mrope writes `cells` rows), copy_to_slot copying the main region into a slot's when copy_needed.
struct Table {
    int64_t stride, cells;
    int slots;
    std::vector<int32_t> tab;
    std::vector<char> slot_img;
    bool main_identity = true;
    int copies = 0;
    Table(int64_t max_context, int n) : stride(mr::stride(max_context)), cells(max_context + 64), slots(n) {
        const std::vector<int32_t> id = mr::identity(stride);
        for (int r = 0; r <= n; ++r) tab.insert(tab.end(), id.begin(), id.end());
        slot_img.assign((size_t) n, 0);
    }
    void parse(const std::vector<int32_t>* img_layout) {   // a GEN (nullptr) or a GENI
        if (img_layout == nullptr && main_identity) return;  // (`geni || !mrope_identity`)
        const std::vector<int32_t> v = img_layout ? *img_layout : mr::identity(cells);
        std::memcpy(tab.data(), v.data(), (size_t) cells * 3 * sizeof(int32_t));
        main_identity = img_layout == nullptr;
    }
    void admit(int b, bool skip_copy = false) {             // copy_to_slot's region part
        if (!mr::copy_needed(main_identity, slot_img[(size_t) b] != 0)) return;
        if (!skip_copy) {
            std::memcpy(tab.data() + (size_t) mr::slot_base(b, stride) * 3, tab.data(),
                        (size_t) stride * 3 * sizeof(int32_t));
            ++copies;
        }
        slot_img[(size_t) b] = main_identity ? 0 : 1;
    }
    // slot b's row for cell c as the rope kernels see it
    int32_t slot_rot(int b, int64_t c, int pair) const { return rot(tab, mr::rope_row(mr::slot_base(b, stride), c), pair); }
};
}  // namespace

int main() {
    // the layout
    check(mr::stride(262144) == 262208, "262K: max_context + 64, a multiple of 64");
    check(mr::stride(1000) % 64 == 0 && mr::stride(1000) >= 1064, "a stride is a multiple of 64 and holds the cells");
    for (int64_t ctx : {1, 4095, 4096, 32768, 131072, 262144, 1000000})
        check(mr::stride(ctx) % 64 == 0 && mr::stride(ctx) >= ctx + 64 && mr::stride(ctx) - (ctx + 64) < 64,
              "stride: the cells rounded up to 64");
    check(mr::slot_base(0, 262208) % 4 == 0, "a base is a valid indexer pos_base (a multiple of 4)");
    // the regions to allocate: only with --vision, --serve and 2 or more slots
    check(mr::slot_regions(true, true, 16) == 16, "16 slots: 16 slot regions");
    check(mr::slot_regions(false, true, 16) == 0, "no --vision: no slot region (every base 0)");
    check(mr::slot_regions(true, false, 16) == 0, "no --serve: no slot region");
    check(mr::slot_regions(true, true, 1) == 0 && mr::slot_regions(true, true, 0) == 0, "fewer than 2 slots: none");
    check(mr::table_entries(16, 262208) == 17LL * 262208 * 3, "17 regions of 262208 rows");
    check(mr::table_entries(16, 262208) * 4 < 52LL * 1048576, "about 51 MiB per GPU at 262K and 16 slots");
    check(mr::table_entries(100000, 262208) == 0, "an index past an int is refused");
    check(mr::table_entries(-1, 64) == 0 && mr::table_entries(2, 0) == 0, "bad arguments are refused");
    // the bases: distinct, region-aligned, never overlapping the main region or each other
    {
        const int64_t st = mr::stride(262144);
        std::set<int32_t> seen;
        bool ok = true;
        for (int b = 0; b < 16; ++b) {
            const int32_t base = mr::slot_base(b, st);
            ok = ok && base == (b + 1) * st && seen.insert(base).second && base >= st;
            // the last cell a slot can reach stays inside its region
            ok = ok && mr::rope_row(base, 262144 + 63) < base + st;
        }
        check(ok, "slot bases: region 1 + b, inside the slot's region, distinct");
        check((int64_t) mr::slot_base(15, st) + st <= mr::table_entries(16, st) / 3, "the last region fits the table");
    }
    // the copy decision
    check(!mr::copy_needed(true, false), "text after text: no copy (text-only traffic never copies)");
    check(mr::copy_needed(false, false), "an image request: the copy");
    check(mr::copy_needed(true, true), "text into a slot whose region still holds an image: the copy (back to identity)");
    check(mr::copy_needed(false, true), "an image into an image slot: the copy");
    // an identity region reads the cell back for every pair: the kernels see the integers of a run without --vision
    {
        Table t(4096, 4);
        bool ok = true;
        for (int b = 0; b < 4; ++b)
            for (int64_t c : {0, 1, 7, 100, 4096, 4159})
                for (int pair = 0; pair < 32; ++pair) ok = ok && t.slot_rot(b, c, pair) == c;
        check(ok, "every region starts as the identity: row(base + c) = c for every pair");
    }
    // the bug and the fix: an image read and decoded while text slots decode
    {
        const int64_t ctx = 4096;
        Table t(ctx, 4);
        const std::vector<int32_t> img = layout(40, {{5, 4, 3}, {20, 2, 5}}, ctx + 64);   // two images, text between
        // two text slots decode (slots 0 and 1)
        t.parse(nullptr); t.admit(0);
        t.parse(nullptr); t.admit(1);
        check(t.copies == 0, "text admissions copy nothing");
        // an image request is parsed (region 0 rebuilt) and read while they decode, then admitted into slot 2
        t.parse(&img);
        bool text_ok = true;
        for (int b : {0, 1})
            for (int64_t c = 0; c < 200; ++c)
                for (int pair = 0; pair < 3; ++pair) text_ok = text_ok && t.slot_rot(b, c, pair) == c;
        check(text_ok, "text slots keep their positions while the image's prompt is read (region 0 is not theirs)");
        t.admit(2);
        check(t.copies == 1 && t.slot_img[2] == 1, "the image admission copies the main region into its slot");
        bool img_ok = true;
        for (int64_t c = 0; c < ctx; ++c)
            for (int pair = 0; pair < 3; ++pair) img_ok = img_ok && t.slot_rot(2, c, pair) == rot(img, c, pair);
        check(img_ok, "the image slot reads the image's layout, the decode tail p + (c - n) included");
        // what that layout is (mtmd): the first image starts at p = 5, so its cell (y = 1, x = 2) is (5, 6, 7)
        check(t.slot_rot(2, 5 + 1 * 4 + 2, 0) == 5 && t.slot_rot(2, 5 + 1 * 4 + 2, 1) == 6 &&
              t.slot_rot(2, 5 + 1 * 4 + 2, 2) == 7, "an image cell is (p, p + y, p + x) in its slot");
        check(t.slot_rot(2, 40, 0) == rot(img, 39, 0) + 1, "the first decode cell continues the text after the prompt");
        // a text request is admitted into slot 3 while the image slot decodes: region 0 back to the identity
        t.parse(nullptr);
        t.admit(3);
        check(t.copies == 1, "a text admission beside an image slot copies nothing (slot 3 is the identity)");
        bool both = true;
        for (int64_t c = 0; c < 200; ++c) {
            both = both && t.slot_rot(3, c, 1) == c && t.slot_rot(0, c, 2) == c;
            both = both && t.slot_rot(2, c, 1) == rot(img, c, 1);
        }
        check(both, "the image slot keeps its layout after region 0 went back to the identity; text slots theirs");
        // a second, different image into slot 0 at once (slot 0's text ended)
        const std::vector<int32_t> img2 = layout(30, {{2, 6, 2}}, ctx + 64);
        t.parse(&img2);
        t.admit(0);
        bool two = true;
        for (int64_t c = 0; c < ctx; ++c)
            two = two && t.slot_rot(0, c, 2) == rot(img2, c, 2) && t.slot_rot(2, c, 2) == rot(img, c, 2);
        check(two, "two different images in two slots, each at its own positions");
        // the image slot ends; text takes slot 2: its region goes back to the identity (the copy of an identity region 0)
        t.parse(nullptr);
        t.admit(2);
        check(t.slot_img[2] == 0, "a text admission into an image slot resets its region");
        bool reset = true;
        for (int64_t c = 0; c < t.stride; ++c) reset = reset && t.slot_rot(2, c, 0) == c && t.slot_rot(2, c, 1) == c;
        check(reset, "the whole region is the identity again (the rows past the main table's cells too)");
        // negative control: an admission that skips the copy leaves the slot with another request's positions
        Table u(ctx, 2);
        u.parse(&img);
        u.admit(1, /*skip_copy=*/true);
        bool wrong = false;
        for (int64_t c = 0; c < 40; ++c) wrong = wrong || u.slot_rot(1, c, 1) != rot(img, c, 1);
        check(wrong, "negative control: without the copy the image slot reads the identity (the test sees it)");
    }
    if (failures != 0) return EXIT_FAILURE;
    std::printf("mrope_regions_test: ok\n");
    return EXIT_SUCCESS;
}
