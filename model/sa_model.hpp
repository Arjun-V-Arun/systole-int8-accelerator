// sa_model.hpp
// Cycle-accurate C++ model of rtl/sa_top.sv.
//
// The model keeps exactly the same registers as the RTL (FSM state, flush
// counter, row index, input skew shift registers and, per PE, the
// accumulator plus the two pass-through registers). Each call to tick()
// is one rising clock edge. outputs() returns what the RTL drives on its
// output ports in the current cycle, before the next edge.
//
// Because the registers match one to one, the testbench can run the RTL
// and this model side by side and compare every output on every cycle,
// including cycles where the input or output side is stalled.

#pragma once

#include <cstdint>
#include <vector>

namespace sa {

enum class State : uint8_t { Load = 0, Flush = 1, Drain = 2 };

struct Inputs {
    bool          in_valid  = false;
    const int8_t* in_a      = nullptr;  // ROWS values, column k of A
    const int8_t* in_b      = nullptr;  // COLS values, row k of B
    bool          in_last   = false;
    bool          out_ready = false;
};

struct Outputs {
    bool                 in_ready  = false;
    bool                 out_valid = false;
    bool                 out_last  = false;
    std::vector<int32_t> out_row;       // COLS values, row row_idx of C
};

struct Counters {
    uint64_t cycles      = 0;  // edges since reset
    uint64_t steps       = 0;  // edges on which the array advanced
    uint64_t useful_macs = 0;  // MACs where both operands came from real data
    uint64_t jobs        = 0;  // completed matrix multiplies
};

class Model {
public:
    // pipe_mac mirrors the PIPE_MAC RTL parameter (product register in every PE).
    Model(int rows, int cols, bool pipe_mac = false);

    void    reset();
    Outputs outputs() const;
    void    tick(const Inputs& in);

    int             rows() const { return rows_; }
    int             cols() const { return cols_; }
    bool            pipe_mac() const { return pipe_; }
    State           state() const { return state_; }
    const Counters& counters() const { return ctr_; }
    int32_t         acc(int r, int c) const { return acc_[idx(r, c)]; }

private:
    int idx(int r, int c) const { return r * cols_ + c; }

    int  rows_, cols_;
    bool pipe_;
    int  flush_steps_;

    // Control registers
    State state_;
    int   flush_cnt_;
    int   row_idx_;

    // Skew shift registers: skew_a_[r] has depth r, skew_b_[c] has depth c.
    // Element 0 is the stage nearest the input. Each stage also carries a
    // "real data" tag that only feeds the utilization counter.
    std::vector<std::vector<int8_t>> skew_a_, skew_b_;
    std::vector<std::vector<bool>>   skew_a_tag_, skew_b_tag_;

    // PE registers
    std::vector<int32_t> acc_;
    std::vector<int32_t> prod_q_;   // only used when pipe_ is set
    std::vector<int8_t>  a_reg_, b_reg_;
    std::vector<bool>    a_tag_, b_tag_;

    Counters ctr_;
};

// Reference: plain triple loop with 32-bit two's complement wrap, the same
// arithmetic the RTL uses.
std::vector<int32_t> reference_matmul(const std::vector<int8_t>& A,
                                      const std::vector<int8_t>& B,
                                      int M, int K, int N);

// Drive one complete job through the model with no stalls and return
// C (rows x cols, row-major) plus the number of cycles taken from the first
// input beat to the last output beat.
std::vector<int32_t> run_job(Model& m, const std::vector<int8_t>& A,
                             const std::vector<int8_t>& B, int K,
                             uint64_t* cycles_out);

}  // namespace sa
