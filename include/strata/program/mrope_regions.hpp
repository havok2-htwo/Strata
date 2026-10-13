// include/strata/program/mrope_regions.hpp - the M-RoPE table's regions with --vision and --batch (header-only: it is
// arithmetic).
//
// The table (strata/kernels/mrope.hpp) is one per device.  With batch slots it holds 1 + B regions of `stride` rows:
// region 0 is the main sessions' (rebuilt at every GEN/GENI, as before), region 1 + b is batch slot b's.  A slot row
// passes slot_base(b) + cell as its rope position, so every rope kernel reads the slot's own region without a kernel
// change.  A region starts as the identity; the main region is copied into slot b's when a request is admitted there
// and either region holds an image layout (copy_needed), so text-only traffic never pays for a copy.
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace strata::program::mrope_regions {

/// Rows per region: the main table's cells (max_context + 64, as the table always had) rounded up to 64.  A multiple
/// of the indexer's block (4) so a slot's base is a valid indexer pos_base (native_qsa_indexer requires pos_base % 4).
inline int64_t stride(int64_t max_context) {
    const int64_t cells = (max_context > 0 ? max_context : 0) + 64;
    return (cells + 63) / 64 * 64;
}

/// The slot regions to allocate: the batch slots that run (after every clamp of --batch), only with --vision and
/// --serve and 2 or more slots; 0 otherwise (then every base stays 0 and the table keeps its one region).
inline int slot_regions(bool vision, bool serve, int batch) {
    return vision && serve && batch >= 2 ? batch : 0;
}

/// The table's int32 entries for `slots` slot regions, or 0 when an index would not fit an int (the kernels index it
/// with int row * 3 + pair).
inline int64_t table_entries(int slots, int64_t stride_rows) {
    if (slots < 0 || stride_rows <= 0) return 0;
    const int64_t rows = (int64_t) (1 + slots) * stride_rows;
    if (rows > std::numeric_limits<int32_t>::max() / 3) return 0;
    return rows * 3;
}

/// The table row of slot b's cell 0 (region 1 + b).  The main sessions' base is 0.
inline int32_t slot_base(int b, int64_t stride_rows) { return (int32_t) ((int64_t) (1 + b) * stride_rows); }

/// The row a slot row passes as its rope position: its base plus its cell.
inline int32_t rope_row(int32_t base, int64_t cell) { return (int32_t) (base + cell); }

/// Whether an admission into a slot copies the main region into the slot's: the main region holds an image layout
/// (the request has pictures), or the slot's region still holds one from an earlier image request (it must become
/// the identity again).  Neither: both are the identity and the copy would change nothing.
inline bool copy_needed(bool main_identity, bool slot_has_layout) { return !main_identity || slot_has_layout; }

/// One region of the identity, `rows` rows of (t, h, w) = (c, c, c): what every region holds at startup.
inline std::vector<int32_t> identity(int64_t rows) {
    std::vector<int32_t> v((size_t) (rows > 0 ? rows : 0) * 3);
    for (int64_t c = 0; c < rows; ++c) v[(size_t) c * 3] = v[(size_t) c * 3 + 1] = v[(size_t) c * 3 + 2] = (int32_t) c;
    return v;
}

}  // namespace strata::program::mrope_regions
