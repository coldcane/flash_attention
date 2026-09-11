// flash_attention_bf16.cpp — bf16 存储 + fp32 计算的 flash attention（桌面/参考版，自带 main）
// 镜像 flash_attention.cpp：RVV 内核(矩阵块乘、online softmax)与 saturn_flash_lean.cpp 逐字一致，
// 只在“内存里的数据格式”上做手脚：
//   Q/K/V/O 以 bf16 (uint16) 存储 → 位解码成 fp32 工作区 → 全 fp32 计算(S/softmax/P·V/归一化)
//   → 结果位编码回 bf16。
// 为什么用位运算转换：Saturn 指令集是 rv64imafd_v，无 Zvfh、无 Zfbfmin，
//   bf16 没有原生向量指令；好在 bf16 = fp32 的高 16 位（指数域同为 8bit，无数阶差），
//   解码就一个 <<16，编码只要一次 RNE 舍入加 —— 纯移位，比 fp16 那套指数搬移简单得多。
// 本文件自带 main：同一份内核跑两次（fp32 参照 vs bf16 存储路径），打印两路 O 的 max|Δ|
//   —— bf16 尾数仅 7bit(相对精度 ~2^-8≈3.9e-3)，误差应比 fp16 大一档左右，可对照看。
// 与 saturn_flash_bf16.cpp 的关系：内核/转换层一字不差；本文件不做裸机计时，只做正确性对比。
// 编译(任意 riscv g++)：riscv64-unknown-linux-gnu-g++ -march=rv64gcv -O2 flash_attention_bf16.cpp -o flash_attention_bf16
//   （若只有 elf 工具链：riscv64-unknown-elf-g++ -march=rv64imafd_v -O2 -c flash_attention_bf16.cpp 仅验编译）

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifndef BENCH_N
#define BENCH_N   32
#define BENCH_D   64     // 与 saturn_flash_lean.cpp 一致；改这里可测其它 d
#define ND        (BENCH_N * BENCH_D)
#endif

// ---------------- bf16 ↔ fp32 位运算转换（无 Zfbfmin；纯移位 + 一次 RNE 加） ----------------
static inline uint32_t f2b(float f){ uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float b2f(uint32_t u){ float f; memcpy(&f, &u, 4); return f; }

// fp32 → bf16：保留高 16 位，RNE 舍低 16 位。inf/nan 特判（防尾数截成 0 变 inf）
static inline uint16_t f32_to_bf16(float x){
    uint32_t f = f2b(x);
    uint32_t s = (f >> 16) & 0x8000u;
    uint32_t e = (f >> 23) & 0xffu;
    uint32_t m = f & 0x7fffffu;
    if(e == 0xffu){                          // inf / nan
        if(m) return (uint16_t)(s | 0x7fc0u);        // nan → 静默 bf16 nan
        return (uint16_t)(s | 0x7f80u);              // ±inf
    }
    uint32_t t = f + 0x7fffu + ((f >> 16) & 1u);     // RNE：加半个ULP再进位（ties→even）
    return (uint16_t)(t >> 16);                      // 高 16 位即 bf16（含符号/指数；低端自然钳 0）
}

// bf16 → fp32：直接左移 16 位。bf16 与 fp32 指数域同宽，无次正规差，全值精确展开
static inline float bf16_to_f32(uint16_t h){ return b2f((uint32_t)h << 16); }

static inline void bf16_to_f32buf(const uint16_t *h, float *f, size_t n){ for(size_t i = 0; i < n; ++i) f[i] = bf16_to_f32(h[i]); }
static inline void f32buf_to_bf16(const float *f, uint16_t *h, size_t n){ for(size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(f[i]); }

// ---------------- 标量数学（无 libm） ----------------
static inline float exp_pade(float x){
    const float c4 = 1.0f/1680.0f, c3 = 5.0f/168.0f, c2 = 9.0f/56.0f, c1 = 1.0f/2.0f, c0 = 1.0f;
    float mol = c0 + x*(c1 + x*(c2 + x*(c3 + x*c4)));
    float den = c0 - x*(c1 - x*(c2 - x*(c3 - x*c4)));
    return mol/den;
}
static inline float my_sqrtf(float x){
    float r = x > 1.0f ? x : 1.0f;
    for(int i = 0; i < 20; ++i) r = 0.5f*(r + x/r);
    return r;
}

// ---------------- RVV exp：(4,4) 帕德，纯浮点，无 vfcvt ----------------
static inline vfloat32m4_t rvv_exp(vfloat32m4_t x, size_t vl){
    const float c4 = 1.0f/1680.0f, c3 = 5.0f/168.0f,
                c2 = 9.0f/56.0f,   c1 = 1.0f/2.0f, c0 = 1.0f;
    const float d1 = -c1, d3 = -c3;
    vfloat32m4_t v;
    v = __riscv_vfmul_vf_f32m4(x, c4, vl);
    v = __riscv_vfadd_vf_f32m4(v, c3, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    v = __riscv_vfadd_vf_f32m4(v, c2, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    v = __riscv_vfadd_vf_f32m4(v, c1, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    vfloat32m4_t mol = __riscv_vfadd_vf_f32m4(v, c0, vl);
    v = __riscv_vfmul_vf_f32m4(x, c4, vl);
    v = __riscv_vfadd_vf_f32m4(v, d3, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    v = __riscv_vfadd_vf_f32m4(v, c2, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    v = __riscv_vfadd_vf_f32m4(v, d1, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    vfloat32m4_t den = __riscv_vfadd_vf_f32m4(v, c0, vl);
    return __riscv_vfdiv_vv_f32m4(mol, den, vl);
}

// ---------------- matmul RVV（4 行寄存器分块） ----------------
static void matmul_rvv(const float *A, const float *B, float *C, size_t M, size_t K, size_t N){
    const size_t TILE = 4;
    size_t vl = __riscv_vsetvl_e32m4(N);
    size_t i = 0;
    for(; i + TILE <= M; i += TILE){
        vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for(size_t k = 0; k < K; ++k){
            const float *b_row = B + k * N;
            vfloat32m4_t b_vec = __riscv_vle32_v_f32m4(b_row, vl);
            acc0 = __riscv_vfmacc_vf_f32m4(acc0, A[(i + 0) * K + k], b_vec, vl);
            acc1 = __riscv_vfmacc_vf_f32m4(acc1, A[(i + 1) * K + k], b_vec, vl);
            acc2 = __riscv_vfmacc_vf_f32m4(acc2, A[(i + 2) * K + k], b_vec, vl);
            acc3 = __riscv_vfmacc_vf_f32m4(acc3, A[(i + 3) * K + k], b_vec, vl);
        }
        __riscv_vse32_v_f32m4(C + (i + 0) * N, acc0, vl);
        __riscv_vse32_v_f32m4(C + (i + 1) * N, acc1, vl);
        __riscv_vse32_v_f32m4(C + (i + 2) * N, acc2, vl);
        __riscv_vse32_v_f32m4(C + (i + 3) * N, acc3, vl);
    }
    for(; i < M; ++i){
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for(size_t k = 0; k < K; ++k){
            const float *b_row = B + k * N;
            vfloat32m4_t b_vec = __riscv_vle32_v_f32m4(b_row, vl);
            acc = __riscv_vfmacc_vf_f32m4(acc, A[i * K + k], b_vec, vl);
        }
        __riscv_vse32_v_f32m4(C + i * N, acc, vl);
    }
}

// ---------------- online softmax ----------------
static vfloat32m4_t online_softmax_rvv(const float *online_input, float &online_max,
                                       float &online_sum, float &alpha, size_t vl){
    const float NEG_INF = -3.402823466e+38f;
    vfloat32m4_t online_vec = __riscv_vle32_v_f32m4(online_input, vl);
    vfloat32m1_t acc  = __riscv_vfmv_s_f_f32m1(NEG_INF, vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m4_f32m1(online_vec, acc, vl);
    float max_block = __riscv_vfmv_f_s_f32m1_f32(vmax);
    float max_new   = (online_max > max_block) ? online_max : max_block;
    alpha = (online_max > NEG_INF) ? exp_pade(online_max - max_new) : 0.0f;
    online_max = max_new;
    online_vec = __riscv_vfsub_vf_f32m4(online_vec, max_new, vl);
    vfloat32m4_t online_output = rvv_exp(online_vec, vl);
    vfloat32m1_t zero_acc = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(online_output, zero_acc, vl);
    online_sum = online_sum * alpha + __riscv_vfmv_f_s_f32m1_f32(vsum);
    return online_output;
}

// ---------------- flash attention（fp32 内核，与 saturn lean 一致） ----------------
static float K_T[BENCH_D * BENCH_N], Kt_j[BENCH_D * 32];
static float S_ij[32 * 32], P_ij[32 * 32];
static float max_q[32], sum_q[32], alpha_row[32];

void flash_attention_rvv(const float *Q, const float *K, const float *V,
                         float *O, size_t N, size_t d){
    const float scale = 1.0f / my_sqrtf((float)d);
    const size_t Br = 32, Bc = 32;
    size_t vl;
    for(size_t io = 0; io < N * d; ++io) O[io] = 0.0f;

    for(size_t k = 0; k < d; ++k)
        for(size_t j = 0; j < N; j += vl){
            vl = __riscv_vsetvl_e32m4(N - j);
            vfloat32m4_t k_col = __riscv_vlse32_v_f32m4(K + j * d + k, sizeof(float) * d, vl);
            __riscv_vse32_v_f32m4(K_T + k * N + j, k_col, vl);
        }

    for(size_t i = 0; i < N; i += Br){
        size_t br = (N - i < Br) ? (N - i) : Br;
        for(size_t a = 0; a < br; ++a){ max_q[a] = -3.402823466e+38f; sum_q[a] = 0.0f; alpha_row[a] = 1.0f; }
        float *O_i = O + i * d;

        for(size_t j = 0; j < N; j += Bc){
            size_t bc = (N - j < Bc) ? (N - j) : Bc;

            for(size_t k = 0; k < d; ++k)
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(K_T + k * N + j + b, vl);
                    __riscv_vse32_v_f32m4(Kt_j + k * bc + b, v, vl);
                }

            matmul_rvv(Q + i * d, Kt_j, S_ij, br, d, bc);

            for(size_t a = 0; a < br * bc; a += vl){
                vl = __riscv_vsetvl_e32m4(br * bc - a);
                vfloat32m4_t s = __riscv_vle32_v_f32m4(S_ij + a, vl);
                s = __riscv_vfmul_vf_f32m4(s, scale, vl);
                __riscv_vse32_v_f32m4(S_ij + a, s, vl);
            }

            for(size_t a = 0; a < br; ++a)
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t p = online_softmax_rvv(S_ij + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);
                    __riscv_vse32_v_f32m4(P_ij + a * bc + b, p, vl);
                }

            for(size_t a = 0; a < br; ++a){
                float *o_row = O_i + a * d;
                for(size_t p = 0; p < d; p += vl){
                    vl = __riscv_vsetvl_e32m4(d - p);
                    vfloat32m4_t o = __riscv_vle32_v_f32m4(o_row + p, vl);
                    o = __riscv_vfmul_vf_f32m4(o, alpha_row[a], vl);
                    __riscv_vse32_v_f32m4(o_row + p, o, vl);
                }
                for(size_t b = 0; b < bc; ++b){
                    float pad = P_ij[a * bc + b];
                    const float *v_row = V + (j + b) * d;
                    for(size_t p = 0; p < d; p += vl){
                        vl = __riscv_vsetvl_e32m4(d - p);
                        vfloat32m4_t o = __riscv_vle32_v_f32m4(o_row + p, vl);
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(v_row + p, vl);
                        o = __riscv_vfmacc_vf_f32m4(o, pad, v, vl);
                        __riscv_vse32_v_f32m4(o_row + p, o, vl);
                    }
                }
            }
        }
        for(size_t a = 0; a < br; ++a)
            for(size_t p = 0; p < d; p += vl){
                vl = __riscv_vsetvl_e32m4(d - p);
                vfloat32m4_t o = __riscv_vle32_v_f32m4(O_i + a * d + p, vl);
                float inv = 1.0f / sum_q[a];
                o = __riscv_vfmul_vf_f32m4(o, inv, vl);
                __riscv_vse32_v_f32m4(O_i + a * d + p, o, vl);
            }
    }
}

// ---------------- main：同一内核，fp32 参照 vs bf16 存储路径，量化误差对比 ----------------
static float Q[ND], K[ND], V[ND];
static float O_ref[ND], O_bf16[ND], Qd[ND], Kd[ND], Vd[ND], O_rt[ND];
static uint16_t Qh[ND], Kh[ND], Vh[ND], Oh[ND];

int main(){
    const size_t N = BENCH_N, d = BENCH_D;

    // 1) 生成 fp32 原始数据
    for(size_t i = 0; i < ND; ++i){
        Q[i] = (float)(i % 10) * 0.1f;
        K[i] = (float)(i % 7) * 0.1f;
        V[i] = (float)(i % 5) * 0.1f;
    }
    flash_attention_rvv(Q, K, V, O_ref, N, d);        // fp32 参照路

    // 2) bf16 存储路径：f32 → bf16(编码) → bf16 → f32(解码) → 内核 → O 编码回 bf16
    f32buf_to_bf16(Q, Qh, ND); f32buf_to_bf16(K, Kh, ND); f32buf_to_bf16(V, Vh, ND);
    bf16_to_f32buf(Qh, Qd, ND); bf16_to_f32buf(Kh, Kd, ND); bf16_to_f32buf(Vh, Vd, ND);
    flash_attention_rvv(Qd, Kd, Vd, O_bf16, N, d);
    f32buf_to_bf16(O_bf16, Oh, ND);                    // 输出编码回 bf16（真正落盘格式）
    bf16_to_f32buf(Oh, O_rt, ND);                      // 读回方便与参照比

    // 3) 量化误差统计
    float md_in = 0.0f, md_e2e = 0.0f;
    for(size_t i = 0; i < ND; ++i){
        float d1 = O_ref[i] - O_bf16[i];   if(d1 < 0) d1 = -d1;   if(d1 > md_in)  md_in  = d1;  // 仅输入量化
        float d2 = O_ref[i] - O_rt[i];     if(d2 < 0) d2 = -d2;   if(d2 > md_e2e) md_e2e = d2;  // 端到端(含输出量化)
    }

    printf("=== RVV flash: fp32 参照 vs bf16 存储/fp32 计算 ===\n");
    printf("N=%u d=%u\n", (unsigned)N, (unsigned)d);
    printf("Qh[0..3] 原始bf16= %04x %04x %04x %04x  解码= %.6f %.6f %.6f %.6f (Q原始 %.6f ...)\n",
           Qh[0], Qh[1], Qh[2], Qh[3],
           (double)Qd[0], (double)Qd[1], (double)Qd[2], (double)Qd[3], (double)Q[0]);
    printf("O_ref[0..3]    = %.6f %.6f %.6f %.6f\n", (double)O_ref[0], (double)O_ref[1], (double)O_ref[2], (double)O_ref[3]);
    printf("O_bf16[0..3]   = %.6f %.6f %.6f %.6f\n", (double)O_bf16[0], (double)O_bf16[1], (double)O_bf16[2], (double)O_bf16[3]);
    printf("Oh[0..3] 输出bf16= %04x %04x %04x %04x\n", Oh[0], Oh[1], Oh[2], Oh[3]);
    printf("max|ΔO| 仅输入量化(bf16 解码)  = %.3e\n", (double)md_in);
    printf("max|ΔO| 端到端(输出也存bf16)   = %.3e\n", (double)md_e2e);
    printf("说明: bf16 尾数 7bit, 相对精度~2^-8≈3.9e-3, 应比 fp16(2^-11) 大一档(约8x)。\n");
    return 0;
}
