| Injected bug | Effect | Result | Caught by |
|---|---|---|---|
| `sign_extension_dropped` | negative products are zero-extended into the accumulator | CAUGHT | scoreboard, lockstep |
| `unsigned_multiply` | operands multiplied as unsigned | CAUGHT | scoreboard, lockstep |
| `pe_ignores_stall` | PEs keep advancing while the array should be stalled | CAUGHT | scoreboard, lockstep |
| `flush_one_short` | last operand never reaches the bottom-right PE | CAUGHT | scoreboard, lockstep |
| `skew_wrong_tap` | rows below row 1 get the wrong delay | CAUGHT | scoreboard, lockstep |
| `no_clear_between_jobs` | accumulators carry over into the next job | CAUGHT | scoreboard, lockstep |
| `drain_ignores_ready` | output rows advance without a handshake | CAUGHT | lockstep, assertion |
| `ready_during_flush` | input accepted while the array is flushing | CAUGHT | scoreboard, lockstep |
| `pipe_missing_flush_step` | pipelined MAC: last product left in the product register | CAUGHT | scoreboard, lockstep |
| `pipe_reg_not_cleared` | pipelined MAC: product register not cleared between jobs | EQUIVALENT | n/a: On the final flush step every PE multiplies zeros (the last real operands have already moved past it), so prod_q is already 0 when clear fires. The clear is redundant in normal operation and kept only as a safety net. |
