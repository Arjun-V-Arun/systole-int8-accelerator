// selftest.cpp
// Checks the C++ model on its own, before it is trusted as a golden model:
//   1. results equal a plain triple-loop matmul for random and corner-case
//      operands across many array shapes and K values,
//   2. the cycle count with no stalls equals K + (R+C-2) + R,
//   3. utilization counter equals R*C*K useful MACs per job.

#include <cstdio>
#include <random>

#include "sa_model.hpp"

int main() {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> val(-128, 127);
    int fails = 0, jobs = 0;

    const int shapes[][2] = {{1, 1}, {1, 4}, {4, 1}, {2, 3}, {3, 3}, {4, 4}, {5, 7}, {8, 8}, {16, 16}};
    const int ks[] = {1, 2, 3, 7, 16, 33, 128};

    for (int pipe = 0; pipe <= 1; ++pipe)
    for (auto& s : shapes) {
        const int R = s[0], C = s[1];
        sa::Model m(R, C, pipe != 0);
        for (int K : ks) {
            for (int mode = 0; mode < 4; ++mode) {
                std::vector<int8_t> A(R * K), B(K * C);
                for (auto& x : A) x = static_cast<int8_t>(mode == 1 ? -128 : mode == 2 ? 127 : val(rng));
                for (auto& x : B) x = static_cast<int8_t>(mode == 1 ? -128 : mode == 2 ? -128 : val(rng));

                const uint64_t macs0 = m.counters().useful_macs;
                uint64_t cycles = 0;
                const auto got = sa::run_job(m, A, B, K, &cycles);
                const auto ref = sa::reference_matmul(A, B, R, K, C);
                const uint64_t want_cycles = static_cast<uint64_t>(K + (R + C - 2 + pipe) + R);
                const uint64_t macs = m.counters().useful_macs - macs0;
                ++jobs;
                if (got != ref) {
                    std::printf("FAIL result  R=%d C=%d K=%d mode=%d pipe=%d\n", R, C, K, mode, pipe);
                    ++fails;
                }
                if (cycles != want_cycles) {
                    std::printf("FAIL cycles  R=%d C=%d K=%d got=%llu want=%llu\n", R, C, K,
                                (unsigned long long)cycles, (unsigned long long)want_cycles);
                    ++fails;
                }
                if (macs != static_cast<uint64_t>(R) * C * K) {
                    std::printf("FAIL macs    R=%d C=%d K=%d got=%llu\n", R, C, K, (unsigned long long)macs);
                    ++fails;
                }
            }
        }
    }
    std::printf("model selftest: %d jobs, %d failures\n", jobs, fails);
    return fails ? 1 : 0;
}
