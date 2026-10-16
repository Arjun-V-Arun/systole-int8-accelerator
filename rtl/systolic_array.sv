// systolic_array.sv
// ROWS x COLS grid of PEs.
//
//            b_edge[0]  b_edge[1]  ...  b_edge[COLS-1]
//               |          |                |
//  a_edge[0] -> PE(0,0) -> PE(0,1) -> ... -> PE(0,COLS-1)
//               |          |                |
//  a_edge[1] -> PE(1,0) -> PE(1,1) -> ... -> PE(1,COLS-1)
//               ...
//
// Operands of A move right one PE per step, operands of B move down one PE
// per step. PE(r,c) owns C[r][c]. The edge inputs must already be skewed
// (row r delayed by r steps, column c delayed by c steps) so that A[r][k]
// and B[k][c] meet in PE(r,c) on the same step, namely step k + r + c.
//
// Ports use flat packed vectors because unpacked array ports are not
// portable across simulators (Icarus and cocotb in particular).

module systolic_array #(
    parameter int ROWS  = 4,
    parameter int COLS  = 4,
    parameter int DW    = 8,
    parameter int ACC_W = 32,
    parameter int PIPE_MAC = 0
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         step,
    input  logic                         clear,
    input  logic [ROWS*DW-1:0]           a_edge,   // a_edge[r] enters row r
    input  logic [COLS*DW-1:0]           b_edge,   // b_edge[c] enters column c
    output logic [ROWS*COLS*ACC_W-1:0]   acc_flat  // C[r][c] at index r*COLS+c
);

    // a_bus[r][c] is the A operand entering PE(r,c); column COLS is the
    // (unused) output of the last PE in the row. Same idea for b_bus.
    logic signed [DW-1:0] a_bus [ROWS][COLS+1];
    logic signed [DW-1:0] b_bus [ROWS+1][COLS];

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : g_a_edge
            assign a_bus[r][0] = a_edge[r*DW +: DW];
        end
        for (c = 0; c < COLS; c++) begin : g_b_edge
            assign b_bus[0][c] = b_edge[c*DW +: DW];
        end

        for (r = 0; r < ROWS; r++) begin : g_row
            for (c = 0; c < COLS; c++) begin : g_col
                pe #(
                    .DW       (DW),
                    .ACC_W    (ACC_W),
                    .PIPE_MAC (PIPE_MAC)
                ) u_pe (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .step  (step),
                    .clear (clear),
                    .a_in  (a_bus[r][c]),
                    .b_in  (b_bus[r][c]),
                    .a_out (a_bus[r][c+1]),
                    .b_out (b_bus[r+1][c]),
                    .acc   (acc_flat[(r*COLS+c)*ACC_W +: ACC_W])
                );
            end
        end
    endgenerate

endmodule
