// pe.sv
// One processing element (PE) of an output-stationary systolic array.
//
// Every step the PE:
//   1. multiplies the operand arriving from the left (a_in) with the operand
//      arriving from above (b_in) and adds the product to its private
//      accumulator (the output element it owns stays put, hence
//      "output-stationary"),
//   2. registers a_in and b_in and passes them on to the right and down.
//
// The PE only advances when `step` is high. When the array is stalled
// (upstream has no data) every register holds its value, which keeps the
// diagonal wavefront aligned.
//
// `clear` zeroes the accumulator and the pass-through registers so a new
// job starts from a clean state. It has priority over `step`.
//
// PIPE_MAC = 1 inserts a register between the multiplier and the adder.
// The product formed on step t is accumulated on step t+1, which splits the
// multiply and the 32-bit add into separate clock cycles (shorter critical
// path, see synth/). The operand flow through the array is unchanged; the
// control only needs one extra flush step to empty the product register.

module pe #(
    parameter int DW    = 8,    // operand width (signed INT8)
    parameter int ACC_W = 32,   // accumulator width
    parameter int PIPE_MAC = 0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    step,
    input  logic                    clear,
    input  logic signed [DW-1:0]    a_in,
    input  logic signed [DW-1:0]    b_in,
    output logic signed [DW-1:0]    a_out,
    output logic signed [DW-1:0]    b_out,
    output logic signed [ACC_W-1:0] acc
);

    // Full-precision signed product: INT8 x INT8 fits in 16 bits.
    logic signed [2*DW-1:0]  prod;
    logic signed [2*DW-1:0]  addend;   // product that gets accumulated this step
    logic signed [ACC_W-1:0] prod_ext;

    assign prod = a_in * b_in;

    generate
        if (PIPE_MAC != 0) begin : g_pipe
            logic signed [2*DW-1:0] prod_q;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      prod_q <= '0;
                else if (clear)  prod_q <= '0;
                else if (step)   prod_q <= prod;
            end
            assign addend = prod_q;
        end else begin : g_comb
            assign addend = prod;
        end
    endgenerate

    // Explicit sign extension to the accumulator width.
    assign prod_ext = {{(ACC_W-2*DW){addend[2*DW-1]}}, addend};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= '0;
            a_out <= '0;
            b_out <= '0;
        end else if (clear) begin
            acc   <= '0;
            a_out <= '0;
            b_out <= '0;
        end else if (step) begin
            acc   <= acc + prod_ext;
            a_out <= a_in;
            b_out <= b_in;
        end
    end

endmodule
