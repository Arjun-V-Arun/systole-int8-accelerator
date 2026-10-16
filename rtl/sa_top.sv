// sa_top.sv
// INT8 matrix-multiply engine: C (ROWS x COLS, INT32) = A (ROWS x K) * B (K x COLS)
//
// Interface (both directions use a valid/ready handshake, a transfer happens
// on a rising clock edge where valid && ready):
//
//   Input stream, K beats per job. Beat k carries
//     in_a = column k of A  (ROWS signed INT8 values, element r at bits [r*8 +: 8])
//     in_b = row k of B     (COLS signed INT8 values, element c at bits [c*8 +: 8])
//     in_last marks beat K-1.
//
//   Output stream, ROWS beats per job. Beat r carries
//     out_row = row r of C  (COLS signed INT32 values, element c at bits [c*32 +: 32])
//     out_last marks row ROWS-1.
//
// Phases (one job at a time, no overlap between jobs):
//
//   LOAD   accept input beats. Each accepted beat advances the array one step.
//          If in_valid is low the whole array stalls, so input bubbles cost
//          time but never corrupt the result.
//   FLUSH  feed zeros for ROWS+COLS-2 steps (one more with PIPE_MAC) so the
//          last operands reach the bottom-right PE and get accumulated.
//   DRAIN  stream out the accumulators row by row. When the last row is
//          accepted the accumulators are cleared and the engine returns to LOAD.
//
// Latency with no stalls, from the first accepted input beat to the last
// output beat: K + (ROWS+COLS-2+PIPE_MAC) + ROWS cycles. The C++ model in model/
// reproduces this cycle for cycle.

module sa_top #(
    parameter int ROWS  = 4,
    parameter int COLS  = 4,
    parameter int DW    = 8,
    parameter int ACC_W = 32,
    parameter int PIPE_MAC = 0   // 1: register between multiplier and adder in every PE
) (
    input  logic                      clk,
    input  logic                      rst_n,

    input  logic                      in_valid,
    output logic                      in_ready,
    input  logic [ROWS*DW-1:0]        in_a,
    input  logic [COLS*DW-1:0]        in_b,
    input  logic                      in_last,

    output logic                      out_valid,
    input  logic                      out_ready,
    output logic [COLS*ACC_W-1:0]     out_row,
    output logic                      out_last
);

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    localparam int FLUSH_STEPS = ROWS + COLS - 2 + PIPE_MAC;
    localparam int FCNT_W      = (FLUSH_STEPS > 1) ? $clog2(FLUSH_STEPS) : 1;
    localparam int RIDX_W      = (ROWS > 1) ? $clog2(ROWS) : 1;

    typedef enum logic [1:0] {
        S_LOAD  = 2'd0,
        S_FLUSH = 2'd1,
        S_DRAIN = 2'd2
    } state_t;

    state_t              state;
    logic [FCNT_W-1:0]   flush_cnt;
    logic [RIDX_W-1:0]   row_idx;

    logic in_fire, out_fire, step, clear;

    assign in_ready  = (state == S_LOAD);
    assign out_valid = (state == S_DRAIN);
    assign out_last  = out_valid && (row_idx == RIDX_W'(ROWS-1));

    assign in_fire   = in_valid  && in_ready;
    assign out_fire  = out_valid && out_ready;

    // The array advances on every accepted input beat and on every flush cycle.
    assign step      = in_fire || (state == S_FLUSH);
    // Clear the datapath as the final output row leaves.
    assign clear     = out_fire && out_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_LOAD;
            flush_cnt <= '0;
            row_idx   <= '0;
        end else begin
            case (state)
                S_LOAD: begin
                    if (in_fire && in_last) begin
                        flush_cnt <= '0;
                        row_idx   <= '0;
                        state     <= (FLUSH_STEPS == 0) ? S_DRAIN : S_FLUSH;
                    end
                end
                S_FLUSH: begin
                    if (flush_cnt == FCNT_W'(FLUSH_STEPS-1)) begin
                        flush_cnt <= '0;
                        state     <= S_DRAIN;
                    end else begin
                        flush_cnt <= flush_cnt + 1'b1;
                    end
                end
                S_DRAIN: begin
                    if (out_fire) begin
                        if (out_last) begin
                            row_idx <= '0;
                            state   <= S_LOAD;
                        end else begin
                            row_idx <= row_idx + 1'b1;
                        end
                    end
                end
                default: state <= S_LOAD;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Input skew
    // Row r of A is delayed by r steps and column c of B by c steps, using a
    // shift register of depth r (or c). During FLUSH zeros are shifted in.
    // ------------------------------------------------------------------
    logic [ROWS*DW-1:0] a_edge;
    logic [COLS*DW-1:0] b_edge;

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : g_skew_a
            logic [DW-1:0] src;
            assign src = in_fire ? in_a[r*DW +: DW] : '0;
            if (r == 0) begin : g_direct
                assign a_edge[0 +: DW] = src;
            end else begin : g_delay
                logic [DW-1:0] sr [r];
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (int i = 0; i < r; i++) sr[i] <= '0;
                    end else if (clear) begin
                        for (int i = 0; i < r; i++) sr[i] <= '0;
                    end else if (step) begin
                        sr[0] <= src;
                        for (int i = 1; i < r; i++) sr[i] <= sr[i-1];
                    end
                end
                assign a_edge[r*DW +: DW] = sr[r-1];
            end
        end

        for (c = 0; c < COLS; c++) begin : g_skew_b
            logic [DW-1:0] src;
            assign src = in_fire ? in_b[c*DW +: DW] : '0;
            if (c == 0) begin : g_direct
                assign b_edge[0 +: DW] = src;
            end else begin : g_delay
                logic [DW-1:0] sr [c];
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (int i = 0; i < c; i++) sr[i] <= '0;
                    end else if (clear) begin
                        for (int i = 0; i < c; i++) sr[i] <= '0;
                    end else if (step) begin
                        sr[0] <= src;
                        for (int i = 1; i < c; i++) sr[i] <= sr[i-1];
                    end
                end
                assign b_edge[c*DW +: DW] = sr[c-1];
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Array and output mux
    // ------------------------------------------------------------------
    logic [ROWS*COLS*ACC_W-1:0] acc_flat;

    systolic_array #(
        .ROWS     (ROWS),
        .COLS     (COLS),
        .DW       (DW),
        .ACC_W    (ACC_W),
        .PIPE_MAC (PIPE_MAC)
    ) u_array (
        .clk      (clk),
        .rst_n    (rst_n),
        .step     (step),
        .clear    (clear),
        .a_edge   (a_edge),
        .b_edge   (b_edge),
        .acc_flat (acc_flat)
    );

    assign out_row = acc_flat[row_idx*COLS*ACC_W +: COLS*ACC_W];

    // ------------------------------------------------------------------
    // Assertions
    // Written as plain checks on registered "previous cycle" values so the
    // same code runs in simulation (ASSERT_ON, Icarus) and in the bounded
    // formal proof (FORMAL, Yosys SAT). See formal/README.md.
    // ------------------------------------------------------------------
`ifdef FORMAL
    `define SA_CHECK(cond, msg) assert (cond);
`else
    // In simulation a failed check prints an error and bumps check_errors,
    // which the cocotb testbench reads at the end of every test.
    `define SA_CHECK(cond, msg) if (!(cond)) begin $error("[sa_top] %s", msg); check_errors <= check_errors + 1; end
`endif

`ifdef SA_CHECKS_ON
    /* verilator lint_off SYNCASYNCNET */
    /* verilator lint_off UNSIGNED */
`ifndef FORMAL
    integer                 check_errors = 0;
`endif
    logic                   past_valid;
    logic                   past_hold;
    logic                   past_clear;
    logic [COLS*ACC_W-1:0]  past_row;
    logic                   past_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            past_valid <= 1'b0;
            past_hold  <= 1'b0;
            past_clear <= 1'b0;
            past_row   <= '0;
            past_last  <= 1'b0;
        end else begin
            past_valid <= 1'b1;
            past_hold  <= out_valid && !out_ready;
            past_clear <= clear;
            past_row   <= out_row;
            past_last  <= out_last;
        end
    end

    always @(posedge clk) begin
        if (rst_n) begin
            `SA_CHECK(state == S_LOAD || state == S_FLUSH || state == S_DRAIN,
                      "illegal FSM state")
            `SA_CHECK({1'b0, row_idx} < (RIDX_W+1)'(ROWS),
                      "row_idx out of range")
            `SA_CHECK(state != S_FLUSH || {1'b0, flush_cnt} < (FCNT_W+1)'(FLUSH_STEPS),
                      "flush counter out of range")
            `SA_CHECK(!(in_ready && out_valid),
                      "input and output phases overlap")
            `SA_CHECK(!out_last || out_valid,
                      "out_last without out_valid")
            `SA_CHECK(state == S_DRAIN || row_idx == '0,
                      "row_idx not reset outside DRAIN")
            if (past_valid && past_hold) begin
                `SA_CHECK(out_valid,            "out_valid dropped before handshake")
                `SA_CHECK(out_row == past_row,  "out_row changed while stalled")
                `SA_CHECK(out_last == past_last,"out_last changed while stalled")
            end
            if (past_valid && past_clear) begin
                `SA_CHECK(acc_flat == '0, "accumulators not zero after clear")
                `SA_CHECK(state == S_LOAD, "not back in LOAD after clear")
            end
        end
    end
    /* verilator lint_on UNSIGNED */
    /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
