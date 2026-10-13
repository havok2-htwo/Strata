#pragma once
// The row layout of a batch window (Verifier::stage_batch), as pure logic so a CPU test can pin it down.
//
// A batch window carries S rows, each a token of one slot.  A slot owns one row (plain --batch) or a contiguous group
// of rows at consecutive positions (--batch-mtp: the slot's current token and its MTP proposal; a layer split's every
// stage sees the same layout).  The rows travel between the stages of a layer split in hand-off buffers of
// `handoff_rows` rows, the window's group starting at row `hbase`.

#include <cstddef>
#include <cstdint>

namespace strata::core {

/// nullptr when the layout is valid, else why not.  `rows[t]` is row t's slot, `pos[t]` its position.  `n_slots`: the
/// slots the verifier holds; `max_t`: its window capacity; `via_handoff`: the stage hands its rows on or takes them from
/// another stage (a layer split), so [hbase, hbase + S) must fit the `handoff_rows` rows of the hand-off buffers.
inline const char* batch_rows_error(const int* rows, int S, const int64_t* pos, int n_slots, int max_t, int hbase,
                                    bool via_handoff, int handoff_rows) {
    if (S < 1 || S > max_t || hbase < 0 || (via_handoff && hbase + S > handoff_rows))
        return "verify: batch rows out of range (init_slots)";
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= n_slots) return "verify: a batch row's slot is out of range";
        for (int u = 0; u < t - 1; ++u)
            if (rows[u] == rows[t] && rows[t - 1] != rows[t]) return "verify: a slot's proposed rows must be contiguous";
        if (t > 0 && rows[t] == rows[t - 1] && pos[t] != pos[t - 1] + 1)
            return "verify: proposed rows must have consecutive positions";
    }
    return nullptr;
}

/// Row t's rope positions in the verifier's staging, [max_t][n_head] q rows, then [max_t][n_head_kv] K rows, then
/// [max_t][idx_q_heads] indexer-query rows: every head of every group at `mrope_base + cell` (--vision with --batch:
/// the slot's region of the M-RoPE table; base 0 otherwise, the cell itself).  The KV cell, the attention length and
/// the QSA step are staged apart (qsa_step_fill) and keep the cell.
inline void batch_rows_stage_rope(int32_t* h_pos, int max_t, int64_t n_head, int64_t n_head_kv, int64_t idx_q_heads,
                                  int t, int64_t cell, int32_t mrope_base) {
    const int32_t rp = (int32_t) (mrope_base + cell);
    for (int64_t h = 0; h < n_head; ++h) h_pos[t * n_head + h] = rp;
    int32_t* pk = h_pos + (size_t) max_t * n_head;
    int32_t* pi = pk + (size_t) max_t * n_head_kv;
    for (int64_t h = 0; h < n_head_kv; ++h) pk[t * n_head_kv + h] = rp;
    for (int64_t h = 0; h < idx_q_heads; ++h) pi[t * idx_q_heads + h] = rp;
}

}  // namespace strata::core
