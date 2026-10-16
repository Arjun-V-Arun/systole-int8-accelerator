## Whole Transformer Block (GPT-2 Small, INT8)

| Phase | Array | OS-built cycles | OS-built util | OS-dbuf util | WS-dbuf util | OS traffic (MB) | WS traffic (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| prefill S=128 | 8x8 | 14954496 | 97.3 % | 100.0 % | 100.0 % | 237.5 | 1050.1 |
| prefill S=128 | 16x16 | 3849216 | 94.5 % | 100.0 % | 100.0 % | 121.1 | 526.3 |
| prefill S=128 | 32x32 | 1017600 | 89.4 % | 99.7 % | 99.7 % | 62.9 | 264.4 |
| prefill S=128 | 64x64 | 282048 | 80.6 % | 97.7 % | 97.7 % | 33.8 | 133.5 |
| decode KV=512 | 8x8 | 1021056 | 12.0 % | 12.5 % | 12.5 % | 8.9 | 16.7 |
| decode KV=512 | 16x16 | 531264 | 5.8 % | 6.2 % | 6.2 % | 8.4 | 12.2 |
| decode KV=512 | 32x32 | 286368 | 2.7 % | 3.1 % | 3.1 % | 8.2 | 10.0 |
| decode KV=512 | 64x64 | 163920 | 1.2 % | 1.5 % | 1.5 % | 8.0 | 8.9 |

## Per Layer, 16x16 Array, prefill S=128

| Layer | M x K x N | Count | OS-built util | OS-dbuf util | WS-dbuf util | OS traffic (MB) | WS traffic (MB) | Share of OS-built time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QKV projection | 128 x 768 x 2304 | 1 | 94.3 % | 100.0 % | 100.0 % | 29.49 | 127.99 | 24.4 % |
| Q x K^T (per head) | 128 x 64 x 128 | 12 | 58.2 % | 98.9 % | 98.9 % | 2.36 | 6.39 | 2.2 % |
| scores x V (per head) | 128 x 128 x 64 | 12 | 73.6 % | 98.9 % | 98.9 % | 1.97 | 6.78 | 1.7 % |
| output projection | 128 x 768 x 768 | 1 | 94.3 % | 100.0 % | 100.0 % | 9.83 | 42.66 | 8.1 % |
| FFN up | 128 x 768 x 3072 | 1 | 94.3 % | 100.0 % | 100.0 % | 39.32 | 170.66 | 32.5 % |
| FFN down | 128 x 3072 x 768 | 1 | 98.5 % | 100.0 % | 100.0 % | 38.14 | 171.84 | 31.1 % |

## Per Layer, 16x16 Array, decode KV=512

| Layer | M x K x N | Count | OS-built util | OS-dbuf util | WS-dbuf util | OS traffic (MB) | WS traffic (MB) | Share of OS-built time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QKV projection | 1 x 768 x 2304 | 1 | 5.9 % | 6.2 % | 6.2 % | 1.89 | 2.76 | 22.1 % |
| q x K^T (per head) | 1 x 64 x 512 | 12 | 3.6 % | 6.1 % | 6.1 % | 0.44 | 0.59 | 8.0 % |
| scores x V (per head) | 1 x 512 x 64 | 12 | 5.7 % | 6.1 % | 6.1 % | 0.42 | 0.61 | 5.0 % |
| output projection | 1 x 768 x 768 | 1 | 5.9 % | 6.2 % | 6.2 % | 0.63 | 0.92 | 7.4 % |
| FFN up | 1 x 768 x 3072 | 1 | 5.9 % | 6.2 % | 6.2 % | 2.52 | 3.67 | 29.4 % |
| FFN down | 1 x 3072 x 768 | 1 | 6.2 % | 6.2 % | 6.2 % | 2.51 | 3.68 | 28.2 % |
