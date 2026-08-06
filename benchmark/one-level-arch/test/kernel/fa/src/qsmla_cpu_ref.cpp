// CPU reference implementation for quant_sparse_flash_mla SWA mode
// Computes: O = softmax(Q @ K^T * softmax_scale + mask) @ V
// Where K=V=ori_kv (MLA shared KV), with token-level sliding window mask

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <iostream>

#ifndef S1
#define S1 64
#endif
#ifndef S2
#define S2 128
#endif
#ifndef D
#define D 512
#endif
#ifndef KTM
#define KTM 32
#endif
#ifndef KTK
#define KTK 32
#endif
#ifndef WIN_LEFT
#define WIN_LEFT 1
#endif
#ifndef WIN_RIGHT
#define WIN_RIGHT 1
#endif
#ifndef SOFTMAX_SCALE
#define SOFTMAX_SCALE 0.125f
#endif

typedef float f32_t;

static void init_deterministic_f32(f32_t* data, int count, int seed) {
    for (int i = 0; i < count; ++i) {
        float val = ((float)((i * 31 + seed * 17) % 100)) / 100.0f - 0.5f;
        data[i] = val;
    }
}

// Build token-level SWA mask (SEL semantics)
// mask[q * s2 + kv] = true  if kv 在窗口外 (被 mask, score 置 -1e30)
//                   = false if kv 在窗口内 (有效, 保持原 score)
// valid: diagonal - win_left <= kv <= diagonal + win_right
//   where diagonal = (s2 - s1) + q
// Apply: score = mask ? -1e30 : score  (TSEL semantics)
static void build_swa_mask(bool* mask, int s1, int s2, int win_left, int win_right) {
    const int causal_offset = s2 - s1;
    for (int q = 0; q < s1; ++q) {
        int diagonal = causal_offset + q;
        int lo = diagonal - win_left;
        int hi = diagonal + win_right;
        for (int kv = 0; kv < s2; ++kv) {
            bool valid = (kv >= lo) && (kv <= hi);
            mask[q * s2 + kv] = !valid;
        }
    }
}

// CPU reference: SWA MLA attention with token-level mask (TSEL semantics)
// Q: [s1, D], KV: [s2, D] (K=V), O: [s1, D]
void cpu_quant_sparse_flash_mla_swa(
    f32_t* out, const f32_t* q, const f32_t* kv,
    const bool* mask,
    int s1, int s2, int d, int kTm, int kTk,
    float softmax_scale, int win_left, int win_right)
{
    const int Qb = (s1 + kTm - 1) / kTm;
    const int Kb = (s2 + kTk - 1) / kTk;

    f32_t* score = (f32_t*)malloc(kTm * kTk * sizeof(f32_t));
    f32_t* p = (f32_t*)malloc(kTm * kTk * sizeof(f32_t));
    f32_t* pv = (f32_t*)malloc(kTm * d * sizeof(f32_t));

    for (int qi = 0; qi < Qb; ++qi) {

        // Pass 1: online softmax with mask
        f32_t row_max[kTm];
        f32_t row_sum[kTm];
        for (int r = 0; r < kTm; ++r) {
            row_max[r] = -1e30f;
            row_sum[r] = 0.0f;
        }

        for (int j = 0; j < Kb; ++j) {
            for (int r = 0; r < kTm; ++r) {
                int q_row = qi * kTm + r;
                if (q_row >= s1) continue;
                for (int c = 0; c < kTk; ++c) {
                    int kv_row = j * kTk + c;
                    if (kv_row >= s2) { score[r * kTk + c] = -1e30f; continue; }
                    f32_t dot = 0.0f;
                    for (int dd = 0; dd < d; ++dd) {
                        dot += q[q_row * d + dd] * kv[kv_row * d + dd];
                    }
                    // Apply mask: TSEL semantics (mask=true → -1e30, mask=false → keep score)
                    f32_t raw_score = dot * softmax_scale;
                    score[r * kTk + c] = mask[q_row * s2 + kv_row] ? -1e30f : raw_score;
                }
            }

            for (int r = 0; r < kTm; ++r) {
                int q_row = qi * kTm + r;
                if (q_row >= s1) continue;

                f32_t local_max = -1e30f;
                for (int c = 0; c < kTk; ++c) {
                    int kv_row = j * kTk + c;
                    if (kv_row >= s2) continue;
                    if (score[r * kTk + c] > local_max)
                        local_max = score[r * kTk + c];
                }

                f32_t new_max = (row_max[r] > local_max) ? row_max[r] : local_max;
                f32_t scale_old = expf(row_max[r] - new_max);
                row_sum[r] *= scale_old;

                for (int c = 0; c < kTk; ++c) {
                    int kv_row = j * kTk + c;
                    if (kv_row >= s2) continue;
                    row_sum[r] += expf(score[r * kTk + c] - new_max);
                }

                row_max[r] = new_max;
            }
        }

        // Pass 2: compute P @ V with mask
        for (int dd = 0; dd < d; ++dd) {
            for (int r = 0; r < kTm; ++r) {
                pv[r * d + dd] = 0.0f;
            }
        }

        for (int j = 0; j < Kb; ++j) {
            for (int r = 0; r < kTm; ++r) {
                int q_row = qi * kTm + r;
                if (q_row >= s1) continue;
                for (int c = 0; c < kTk; ++c) {
                    int kv_row = j * kTk + c;
                    if (kv_row >= s2) { p[r * kTk + c] = 0.0f; continue; }
                    f32_t dot = 0.0f;
                    for (int dd = 0; dd < d; ++dd) {
                        dot += q[q_row * d + dd] * kv[kv_row * d + dd];
                    }
                    // Apply mask: TSEL semantics (mask=true → -1e30, mask=false → keep score)
                    f32_t raw_score = dot * softmax_scale;
                    f32_t s = mask[q_row * s2 + kv_row] ? -1e30f : raw_score;
                    p[r * kTk + c] = expf(s - row_max[r]) / row_sum[r];
                }
            }

            for (int r = 0; r < kTm; ++r) {
                int q_row = qi * kTm + r;
                if (q_row >= s1) continue;
                for (int dd = 0; dd < d; ++dd) {
                    for (int c = 0; c < kTk; ++c) {
                        int kv_row = j * kTk + c;
                        if (kv_row >= s2) continue;
                        pv[r * d + dd] += p[r * kTk + c] * kv[kv_row * d + dd];
                    }
                }
            }
        }

        for (int r = 0; r < kTm; ++r) {
            int q_row = qi * kTm + r;
            if (q_row >= s1) continue;
            for (int dd = 0; dd < d; ++dd) {
                out[q_row * d + dd] = pv[r * d + dd];
            }
        }
    }

    free(score);
    free(p);
    free(pv);
}

int main() {
    int q_count = S1 * D;
    int kv_count = S2 * D;
    int out_count = S1 * D;
    int mask_count = S1 * S2;

    f32_t* q = (f32_t*)malloc(q_count * sizeof(f32_t));
    f32_t* kv = (f32_t*)malloc(kv_count * sizeof(f32_t));
    f32_t* out = (f32_t*)malloc(out_count * sizeof(f32_t));
    bool* mask = (bool*)malloc(mask_count * sizeof(bool));

    init_deterministic_f32(q, q_count, 1);
    init_deterministic_f32(kv, kv_count, 2);

    // for (int i = 0; i < 100; i++){
    //     std::cout << q[i] << ", ";
    // }

    // std::cout << std::endl;

    // for (int i = 0; i < 100; i++){
    //     std::cout << kv[i] << ", ";
    // }

    // return 0;

    memset(out, 0, out_count * sizeof(f32_t));

    // f32_t mask1[5][10];

    // build_swa_mask(&mask1[0][0], 5, 10, 2, 2);

    // for(int i = 0; i < 5; i++){
    //     for(int j = 0; j < 10; j++){
    //         std::cout << mask1[i][j] << ",       \t";
    //     }
    //     std::cout << std::endl;
    // }

    // return 0;


    build_swa_mask(mask, S1, S2, WIN_LEFT, WIN_RIGHT);

    printf("QSMLA_CPU: s1=%d s2=%d D=%d win_left=%d win_right=%d\n",
           S1, S2, D, WIN_LEFT, WIN_RIGHT);
    printf("QSMLA_CPU: computing reference with token-level mask...\n");
    fflush(stdout);

    cpu_quant_sparse_flash_mla_swa(out, q, kv, mask,
        S1, S2, D, KTM, KTK,
        SOFTMAX_SCALE, WIN_LEFT, WIN_RIGHT);

    const char* golden_path = "qsmla_golden.bin";
    FILE* f = fopen(golden_path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", golden_path);
        return 1;
    }
    fwrite(out, sizeof(f32_t), out_count, f);
    fclose(f);

    printf("QSMLA_CPU: golden output written to %s (%d floats, %d bytes)\n",
           golden_path, out_count, (int)(out_count * sizeof(f32_t)));

    printf("QSMLA_CPU: first 8 output values:\n");
    for (int i = 0; i < 8 && i < out_count; ++i) {
        printf("  out[%d] = %.6f\n", i, out[i]);
    }

    FILE* fq = fopen("qsmla_input_q.bin", "wb");
    fwrite(q, sizeof(f32_t), q_count, fq);
    fclose(fq);

    FILE* fkv = fopen("qsmla_input_kv.bin", "wb");
    fwrite(kv, sizeof(f32_t), kv_count, fkv);
    fclose(fkv);

    printf("QSMLA_CPU: input data written to qsmla_input_q.bin and qsmla_input_kv.bin\n");

    free(q);
    free(kv);
    free(out);
    free(mask);
    return 0;
}
