// sa_model.cpp
// Implementation of the cycle-accurate model plus a C interface (sa_*) so the
// Python testbench can load it with ctypes and run it in lockstep with the RTL.

#include "sa_model.hpp"

#include <algorithm>
#include <stdexcept>

namespace sa {

namespace {
// 32-bit two's complement add, matching a Verilog 32-bit adder that wraps.
inline int32_t wrap_add(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}
}  // namespace

Model::Model(int rows, int cols, bool pipe_mac)
    : rows_(rows), cols_(cols), pipe_(pipe_mac), flush_steps_(rows + cols - 2 + (pipe_mac ? 1 : 0)) {
    if (rows < 1 || cols < 1) throw std::invalid_argument("rows and cols must be >= 1");
    reset();
}

void Model::reset() {
    state_     = State::Load;
    flush_cnt_ = 0;
    row_idx_   = 0;

    skew_a_.assign(rows_, {});
    skew_a_tag_.assign(rows_, {});
    for (int r = 0; r < rows_; ++r) {
        skew_a_[r].assign(r, 0);
        skew_a_tag_[r].assign(r, false);
    }
    skew_b_.assign(cols_, {});
    skew_b_tag_.assign(cols_, {});
    for (int c = 0; c < cols_; ++c) {
        skew_b_[c].assign(c, 0);
        skew_b_tag_[c].assign(c, false);
    }

    acc_.assign(rows_ * cols_, 0);
    prod_q_.assign(rows_ * cols_, 0);
    a_reg_.assign(rows_ * cols_, 0);
    b_reg_.assign(rows_ * cols_, 0);
    a_tag_.assign(rows_ * cols_, false);
    b_tag_.assign(rows_ * cols_, false);
    ctr_ = Counters{};
}

Outputs Model::outputs() const {
    Outputs o;
    o.in_ready  = (state_ == State::Load);
    o.out_valid = (state_ == State::Drain);
    o.out_last  = o.out_valid && (row_idx_ == rows_ - 1);
    o.out_row.resize(cols_);
    // The RTL output mux is always active, so mirror it even when not valid.
    for (int c = 0; c < cols_; ++c) o.out_row[c] = acc_[idx(row_idx_, c)];
    return o;
}

void Model::tick(const Inputs& in) {
    const Outputs o = outputs();

    const bool in_fire  = in.in_valid && o.in_ready;
    const bool out_fire = o.out_valid && in.out_ready;
    const bool step     = in_fire || (state_ == State::Flush);
    const bool clear    = out_fire && o.out_last;

    ++ctr_.cycles;

    // ---- Datapath -------------------------------------------------------
    if (clear) {
        for (int r = 0; r < rows_; ++r) {
            std::fill(skew_a_[r].begin(), skew_a_[r].end(), 0);
            std::fill(skew_a_tag_[r].begin(), skew_a_tag_[r].end(), false);
        }
        for (int c = 0; c < cols_; ++c) {
            std::fill(skew_b_[c].begin(), skew_b_[c].end(), 0);
            std::fill(skew_b_tag_[c].begin(), skew_b_tag_[c].end(), false);
        }
        std::fill(acc_.begin(), acc_.end(), 0);
        std::fill(prod_q_.begin(), prod_q_.end(), 0);
        std::fill(a_reg_.begin(), a_reg_.end(), 0);
        std::fill(b_reg_.begin(), b_reg_.end(), 0);
        std::fill(a_tag_.begin(), a_tag_.end(), false);
        std::fill(b_tag_.begin(), b_tag_.end(), false);
        ++ctr_.jobs;
    } else if (step) {
        ++ctr_.steps;

        // Values entering the array edges this cycle (combinational in RTL).
        std::vector<int8_t> a_edge(rows_), b_edge(cols_);
        std::vector<bool>   a_edge_tag(rows_), b_edge_tag(cols_);
        for (int r = 0; r < rows_; ++r) {
            const int8_t src = in_fire ? in.in_a[r] : 0;
            if (r == 0) {
                a_edge[r] = src;
                a_edge_tag[r] = in_fire;
            } else {
                a_edge[r] = skew_a_[r][r - 1];
                a_edge_tag[r] = skew_a_tag_[r][r - 1];
            }
        }
        for (int c = 0; c < cols_; ++c) {
            const int8_t src = in_fire ? in.in_b[c] : 0;
            if (c == 0) {
                b_edge[c] = src;
                b_edge_tag[c] = in_fire;
            } else {
                b_edge[c] = skew_b_[c][c - 1];
                b_edge_tag[c] = skew_b_tag_[c][c - 1];
            }
        }

        // PE update. Compute every next value from current values first,
        // then commit, exactly like non-blocking assignments.
        std::vector<int8_t> na(rows_ * cols_), nb(rows_ * cols_);
        std::vector<bool>   nat(rows_ * cols_), nbt(rows_ * cols_);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                const int    i     = idx(r, c);
                const int8_t a_in  = (c == 0) ? a_edge[r] : a_reg_[idx(r, c - 1)];
                const int8_t b_in  = (r == 0) ? b_edge[c] : b_reg_[idx(r - 1, c)];
                const bool   at_in = (c == 0) ? a_edge_tag[r] : a_tag_[idx(r, c - 1)];
                const bool   bt_in = (r == 0) ? b_edge_tag[c] : b_tag_[idx(r - 1, c)];
                const int32_t prod = static_cast<int32_t>(a_in) * static_cast<int32_t>(b_in);
                if (pipe_) {
                    acc_[i]    = wrap_add(acc_[i], prod_q_[i]);  // product from the previous step
                    prod_q_[i] = prod;
                } else {
                    acc_[i] = wrap_add(acc_[i], prod);
                }
                if (at_in && bt_in) ++ctr_.useful_macs;
                na[i] = a_in;  nat[i] = at_in;
                nb[i] = b_in;  nbt[i] = bt_in;
            }
        }
        a_reg_ = na;  a_tag_ = nat;
        b_reg_ = nb;  b_tag_ = nbt;

        // Shift the skew registers (stage 0 takes the new input).
        for (int r = 1; r < rows_; ++r) {
            for (int s = r - 1; s > 0; --s) {
                skew_a_[r][s] = skew_a_[r][s - 1];
                skew_a_tag_[r][s] = skew_a_tag_[r][s - 1];
            }
            skew_a_[r][0] = in_fire ? in.in_a[r] : 0;
            skew_a_tag_[r][0] = in_fire;
        }
        for (int c = 1; c < cols_; ++c) {
            for (int s = c - 1; s > 0; --s) {
                skew_b_[c][s] = skew_b_[c][s - 1];
                skew_b_tag_[c][s] = skew_b_tag_[c][s - 1];
            }
            skew_b_[c][0] = in_fire ? in.in_b[c] : 0;
            skew_b_tag_[c][0] = in_fire;
        }
    }

    // ---- Control --------------------------------------------------------
    switch (state_) {
        case State::Load:
            if (in_fire && in.in_last) {
                flush_cnt_ = 0;
                row_idx_   = 0;
                state_     = (flush_steps_ == 0) ? State::Drain : State::Flush;
            }
            break;
        case State::Flush:
            if (flush_cnt_ == flush_steps_ - 1) {
                flush_cnt_ = 0;
                state_     = State::Drain;
            } else {
                ++flush_cnt_;
            }
            break;
        case State::Drain:
            if (out_fire) {
                if (o.out_last) {
                    row_idx_ = 0;
                    state_   = State::Load;
                } else {
                    ++row_idx_;
                }
            }
            break;
    }
}

std::vector<int32_t> reference_matmul(const std::vector<int8_t>& A,
                                      const std::vector<int8_t>& B,
                                      int M, int K, int N) {
    std::vector<int32_t> C(static_cast<size_t>(M) * N, 0);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t s = 0;
            for (int k = 0; k < K; ++k)
                s = wrap_add(s, static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]));
            C[i * N + j] = s;
        }
    return C;
}

std::vector<int32_t> run_job(Model& m, const std::vector<int8_t>& A,
                             const std::vector<int8_t>& B, int K,
                             uint64_t* cycles_out) {
    const int R = m.rows(), Cn = m.cols();
    std::vector<int8_t>  col(R), row(Cn);
    std::vector<int32_t> C;
    C.reserve(static_cast<size_t>(R) * Cn);

    const uint64_t start = m.counters().cycles;
    int k = 0;
    bool done = false;
    while (!done) {
        Inputs in;
        in.out_ready = true;
        if (k < K) {
            for (int r = 0; r < R; ++r) col[r] = A[r * K + k];
            for (int c = 0; c < Cn; ++c) row[c] = B[k * Cn + c];
            in.in_valid = true;
            in.in_a     = col.data();
            in.in_b     = row.data();
            in.in_last  = (k == K - 1);
        }
        const Outputs o = m.outputs();
        if (in.in_valid && o.in_ready) ++k;
        if (o.out_valid) {
            C.insert(C.end(), o.out_row.begin(), o.out_row.end());
            done = o.out_last;
        }
        m.tick(in);
    }
    if (cycles_out) *cycles_out = m.counters().cycles - start;
    return C;
}

}  // namespace sa

// ---------------------------------------------------------------------------
// C interface for ctypes (tb/model_bridge.py)
// ---------------------------------------------------------------------------
extern "C" {

void* sa_new(int rows, int cols, int pipe_mac) { return new sa::Model(rows, cols, pipe_mac != 0); }
void  sa_free(void* h) { delete static_cast<sa::Model*>(h); }
void  sa_reset(void* h) { static_cast<sa::Model*>(h)->reset(); }

void sa_outputs(void* h, int* in_ready, int* out_valid, int* out_last, int32_t* out_row) {
    const sa::Outputs o = static_cast<sa::Model*>(h)->outputs();
    *in_ready  = o.in_ready;
    *out_valid = o.out_valid;
    *out_last  = o.out_last;
    for (size_t c = 0; c < o.out_row.size(); ++c) out_row[c] = o.out_row[c];
}

void sa_tick(void* h, int in_valid, const int8_t* in_a, const int8_t* in_b,
             int in_last, int out_ready) {
    sa::Inputs in;
    in.in_valid  = in_valid != 0;
    in.in_a      = in_a;
    in.in_b      = in_b;
    in.in_last   = in_last != 0;
    in.out_ready = out_ready != 0;
    static_cast<sa::Model*>(h)->tick(in);
}

void sa_counters(void* h, uint64_t* cycles, uint64_t* steps, uint64_t* useful_macs, uint64_t* jobs) {
    const sa::Counters& c = static_cast<sa::Model*>(h)->counters();
    *cycles = c.cycles;
    *steps = c.steps;
    *useful_macs = c.useful_macs;
    *jobs = c.jobs;
}

}  // extern "C"
