// perf_model.cpp
// Architecture study: how well does a systolic array of a given size and
// dataflow serve the GEMMs inside one transformer block?
//
// Workload: one GPT-2 small block (d_model 768, 12 heads of 64, d_ff 3072)
//   prefill: S = 128 tokens processed together
//   decode:  1 new token attending to a 512-token KV cache
//
// Dataflows compared (C = A x B, A is M x K, B is K x N, array is R x Cn):
//
//   OS-built   Output-stationary, exactly as in rtl/sa_top.sv. Each R x Cn
//              output tile streams K beats, flushes R+Cn-2 cycles and drains
//              R rows, with no overlap between tiles. Per-tile cycles are
//              measured by running the cycle-accurate model, not by formula.
//   OS-dbuf    Output-stationary with double-buffered accumulators: the
//              drain of one tile hides under the next tile's stream, and the
//              skew fill/flush is paid once per GEMM. Per tile: max(K, R).
//   WS-dbuf    Weight-stationary (B tile of R x Cn held in the PEs, rows of
//              A streamed through), weights double-buffered so the next
//              preload hides under the current stream. Per weight tile:
//              max(M, R), plus fill/flush once per GEMM. Partial sums across
//              K tiles are assumed to accumulate for free outside the array.
//
// Utilization = useful MACs / (cycles * R * Cn).
//
// On-chip buffer traffic (bytes moved between the operand/result buffers and
// the array edge; INT8 operands are 1 byte, partial sums and results 4 bytes):
//   OS  A is re-read once per column of output tiles, B once per row of
//       output tiles, every result is written once:
//         M*K*ceil(N/Cn) + K*N*ceil(M/R) + 4*M*N
//   WS  every weight is loaded once, A is re-read once per column of weight
//       tiles, and partial sums are written per K tile and read back for all
//       but the first:
//         K*N + M*K*ceil(N/Cn) + 4*M*N*(2*ceil(K/R) - 1)
// This is where the two dataflows really differ: WS saves weight traffic,
// OS keeps the partial sums inside the PEs.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "sa_model.hpp"

struct Gemm {
    std::string name;
    long M, K, N;
    int count;  // e.g. 12 heads
};

static long ceil_div(long a, long b) { return (a + b - 1) / b; }

// Measured with the cycle-accurate model (cached per (R, C, K)).
static long os_built_tile_cycles(int R, int C, long K) {
    static std::map<std::tuple<int, int, long>, long> cache;
    auto key = std::make_tuple(R, C, K);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> v(-128, 127);
    std::vector<int8_t> A(static_cast<size_t>(R) * K), B(static_cast<size_t>(K) * C);
    for (auto& x : A) x = static_cast<int8_t>(v(rng));
    for (auto& x : B) x = static_cast<int8_t>(v(rng));
    sa::Model m(R, C);
    uint64_t cycles = 0;
    const auto got = sa::run_job(m, A, B, static_cast<int>(K), &cycles);
    if (got != sa::reference_matmul(A, B, R, static_cast<int>(K), C)) {
        std::fprintf(stderr, "model result mismatch in perf run\n");
        std::exit(1);
    }
    cache[key] = static_cast<long>(cycles);
    return static_cast<long>(cycles);
}

struct Result {
    double cycles = 0, macs = 0;
};

static double os_bytes(const Gemm& g, int R, int C) {
    return g.count * (double(g.M) * g.K * ceil_div(g.N, C) + double(g.K) * g.N * ceil_div(g.M, R) + 4.0 * g.M * g.N);
}

static double ws_bytes(const Gemm& g, int R, int C) {
    return g.count * (double(g.K) * g.N + double(g.M) * g.K * ceil_div(g.N, C) +
                      4.0 * g.M * g.N * (2.0 * ceil_div(g.K, R) - 1));
}

static Result os_built(const Gemm& g, int R, int C) {
    const long tiles = ceil_div(g.M, R) * ceil_div(g.N, C);
    return {static_cast<double>(g.count) * tiles * os_built_tile_cycles(R, C, g.K),
            static_cast<double>(g.count) * g.M * g.K * g.N};
}

static Result os_dbuf(const Gemm& g, int R, int C) {
    const long tiles = ceil_div(g.M, R) * ceil_div(g.N, C);
    const double per_gemm = tiles * std::max(g.K, static_cast<long>(R)) + (R + C - 2) + R;
    return {g.count * per_gemm, static_cast<double>(g.count) * g.M * g.K * g.N};
}

static Result ws_dbuf(const Gemm& g, int R, int C) {
    const long tiles = ceil_div(g.K, R) * ceil_div(g.N, C);
    const double per_gemm = tiles * std::max(g.M, static_cast<long>(R)) + R + (R + C - 2);
    return {g.count * per_gemm, static_cast<double>(g.count) * g.M * g.K * g.N};
}

static std::vector<Gemm> prefill(long S) {
    return {{"QKV projection", S, 768, 2304, 1},
            {"Q x K^T (per head)", S, 64, S, 12},
            {"scores x V (per head)", S, S, 64, 12},
            {"output projection", S, 768, 768, 1},
            {"FFN up", S, 768, 3072, 1},
            {"FFN down", S, 3072, 768, 1}};
}

static std::vector<Gemm> decode(long L) {
    return {{"QKV projection", 1, 768, 2304, 1},
            {"q x K^T (per head)", 1, 64, L, 12},
            {"scores x V (per head)", 1, L, 64, 12},
            {"output projection", 1, 768, 768, 1},
            {"FFN up", 1, 768, 3072, 1},
            {"FFN down", 1, 3072, 768, 1}};
}

int main(int argc, char** argv) {
    const char* csv_path = argc > 1 ? argv[1] : nullptr;
    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "phase,array,layer,dataflow,cycles,utilization\n");

    const int sizes[] = {8, 16, 32, 64};
    const struct { const char* name; std::vector<Gemm> g; } phases[] = {
        {"prefill S=128", prefill(128)}, {"decode KV=512", decode(512)}};

    // Summary table: whole block
    std::printf("## Whole Transformer Block (GPT-2 Small, INT8)\n\n");
    std::printf("| Phase | Array | OS-built cycles | OS-built util | OS-dbuf util | WS-dbuf util | OS traffic (MB) | WS traffic (MB) |\n");
    std::printf("|---|---|---:|---:|---:|---:|---:|---:|\n");
    for (const auto& ph : phases) {
        for (int n : sizes) {
            Result a{}, b{}, c{};
            double osb = 0, wsb = 0;
            for (const auto& g : ph.g) {
                osb += os_bytes(g, n, n);
                wsb += ws_bytes(g, n, n);
                Result x = os_built(g, n, n), y = os_dbuf(g, n, n), z = ws_dbuf(g, n, n);
                a.cycles += x.cycles; a.macs += x.macs;
                b.cycles += y.cycles; b.macs += y.macs;
                c.cycles += z.cycles; c.macs += z.macs;
                if (csv) {
                    const double pe = static_cast<double>(n) * n;
                    std::fprintf(csv, "%s,%dx%d,%s,OS-built,%.0f,%.4f\n", ph.name, n, n, g.name.c_str(), x.cycles, x.macs / (x.cycles * pe));
                    std::fprintf(csv, "%s,%dx%d,%s,OS-dbuf,%.0f,%.4f\n", ph.name, n, n, g.name.c_str(), y.cycles, y.macs / (y.cycles * pe));
                    std::fprintf(csv, "%s,%dx%d,%s,WS-dbuf,%.0f,%.4f\n", ph.name, n, n, g.name.c_str(), z.cycles, z.macs / (z.cycles * pe));
                }
            }
            const double pe = static_cast<double>(n) * n;
            std::printf("| %s | %dx%d | %.0f | %.1f %% | %.1f %% | %.1f %% | %.1f | %.1f |\n", ph.name, n, n, a.cycles,
                        100 * a.macs / (a.cycles * pe), 100 * b.macs / (b.cycles * pe), 100 * c.macs / (c.cycles * pe),
                        osb / 1e6, wsb / 1e6);
        }
    }

    // Per-layer breakdown for one array size
    const int n = 16;
    for (const auto& ph : phases) {
        std::printf("\n## Per Layer, %dx%d Array, %s\n\n", n, n, ph.name);
        std::printf("| Layer | M x K x N | Count | OS-built util | OS-dbuf util | WS-dbuf util | OS traffic (MB) | WS traffic (MB) | Share of OS-built time |\n");
        std::printf("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
        double total = 0;
        for (const auto& g : ph.g) total += os_built(g, n, n).cycles;
        for (const auto& g : ph.g) {
            Result x = os_built(g, n, n), y = os_dbuf(g, n, n), z = ws_dbuf(g, n, n);
            const double pe = static_cast<double>(n) * n;
            std::printf("| %s | %ld x %ld x %ld | %d | %.1f %% | %.1f %% | %.1f %% | %.2f | %.2f | %.1f %% |\n", g.name.c_str(),
                        g.M, g.K, g.N, g.count, 100 * x.macs / (x.cycles * pe), 100 * y.macs / (y.cycles * pe),
                        100 * z.macs / (z.cycles * pe), os_bytes(g, n, n) / 1e6, ws_bytes(g, n, n) / 1e6,
                        100 * x.cycles / total);
        }
    }
    if (csv) std::fclose(csv);
    return 0;
}
