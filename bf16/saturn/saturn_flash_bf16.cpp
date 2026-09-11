// saturn_flash_bf16.cpp — bf16 存储 + fp32 计算的 flash attention（Saturn 裸机计时版）
// 派生自 saturn_flash_lean.cpp：数据格式层改成 bf16。
//   存储格式：Q/K/V/O 按 bf16 (uint16) 存在内存里（内存带宽减半）
//   计算格式：读回后位解码成 fp32，S/P/O/归一化全在 fp32 算（同 FlashAttention 混合精度）
//   bf16↔fp32：纯移位（bf16 = fp32 的高 16 位，指数域同宽、无数阶差，inf/nan/0 直接扩位），
//               完全无浮点指令参与 —— 唯一要做的只是 fp32→bf16 的 RNE 舍入。
//   与现有内核 rv64imafd_v 指令集一致，不依赖 Zfbfmin/Zvfbfwma。
// 运行: 与 saturn_flash_lean.md 相同流程，换成 saturn_flash_bf16.riscv。
// 正确性参考: 桌面版 flash_attention_bf16.cpp 内有 fp32-vs-bf16 对照。

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define BENCH_N   32
#define BENCH_D   64     // 与 saturn_flash_lean.cpp 当前一致；要复现 D=256 卡死诊断场景改回 256
#define BENCH_BR  32
#define BENCH_BC  32
#define ND        (BENCH_N * BENCH_D)   // 2048

// ---------------- M-mode 性能计数器 ----------------
static inline uint64_t rdcycle(void)   { uint64_t c; asm volatile("csrr %0, mcycle"   : "=r"(c)); return c; }
static inline uint64_t rdinstret(void) { uint64_t c; asm volatile("csrr %0, minstret" : "=r"(c)); return c; }

// ---------------- 分阶段性能统计（测各算子 cycles/instret，g_prof=1 才统计） ----------------
static int g_prof = 0;
static uint64_t c_kt, i_kt;             // K 转置
static uint64_t c_cp, i_cp;             // Kt_j 拷贝
static uint64_t c_mm, i_mm;             // matmul
static uint64_t c_sc, i_sc;             // S *= scale
static uint64_t c_sm, i_sm;             // online softmax（含 exp）
static uint64_t c_oa, i_oa;             // O 修正 + P@V 累加
static uint64_t c_nm, i_nm;             // 归一化
static uint64_t c_exp, i_exp;           // 纯 rvv_exp
static uint64_t c_ep, i_ep;             // 标量 exp_pade（alpha）

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

// bf16 → fp32：直接左移 16 位。bf16 与 fp32 指数域同宽(8bit)，无次正规差，全值精确展开
static inline float bf16_to_f32(uint16_t h){ return b2f((uint32_t)h << 16); }

static inline void bf16_to_f32buf(const uint16_t *h, float *f, size_t n){ for(size_t i = 0; i < n; ++i) f[i] = bf16_to_f32(h[i]); }
static inline void f32buf_to_bf16(const float *f, uint16_t *h, size_t n){ for(size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(f[i]); }

// ---------------- 标量数学（无 libm） ----------------
static inline float exp_pade(float x){
    uint64_t c0m = 0, i0m = 0;
    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }
    const float c4 = 1.0f/1680.0f, c3 = 5.0f/168.0f, c2 = 9.0f/56.0f, c1 = 1.0f/2.0f, c0 = 1.0f;
    float mol = c0 + x*(c1 + x*(c2 + x*(c3 + x*c4)));
    float den = c0 - x*(c1 - x*(c2 - x*(c3 - x*c4)));
    float r = mol/den;
    if(g_prof){ c_ep += rdcycle()-c0m; i_ep += rdinstret()-i0m; }
    return r;
}
static inline float my_sqrtf(float x){
    float r = x > 1.0f ? x : 1.0f;
    for(int i = 0; i < 20; ++i) r = 0.5f*(r + x/r);
    return r;
}

// ---------------- RVV exp：(4,4) 帕德，纯浮点（与 lean 一致，无 vfcvt） ----------------
static inline vfloat32m4_t rvv_exp(vfloat32m4_t x, size_t vl){
    uint64_t c0m = 0, i0m = 0;
    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }
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
    vfloat32m4_t r = __riscv_vfdiv_vv_f32m4(mol, den, vl);
    if(g_prof){ c_exp += rdcycle()-c0m; i_exp += rdinstret()-i0m; }
    return r;
}

// ---------------- matmul RVV（4 行寄存器分块，与 lean 一致） ----------------
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

// ---------------- online softmax（fp32 内部计算） ----------------
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

// ---------------- flash attention（fp32 内核；Q/K/V 已由调用方解码成 fp32） ----------------
static float K_T[BENCH_D * BENCH_N], Kt_j[BENCH_D * BENCH_BC];
static float S_ij[BENCH_BR * BENCH_BC], P_ij[BENCH_BR * BENCH_BC];
static float max_q[BENCH_BR], sum_q[BENCH_BR], alpha_row[BENCH_BR];

void flash_attention_rvv(const float *Q, const float *K, const float *V,
                         float *O, size_t N, size_t d){
    const float scale = 1.0f / my_sqrtf((float)d);
    const size_t Br = 32, Bc = 32;
    size_t vl;
    uint64_t c0m = 0, i0m = 0;
    for(size_t io = 0; io < N * d; ++io) O[io] = 0.0f;

    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }     // K 转置
    for(size_t k = 0; k < d; ++k)
        for(size_t j = 0; j < N; j += vl){
            vl = __riscv_vsetvl_e32m4(N - j);
            vfloat32m4_t k_col = __riscv_vlse32_v_f32m4(K + j * d + k, sizeof(float) * d, vl);
            __riscv_vse32_v_f32m4(K_T + k * N + j, k_col, vl);
        }
    if(g_prof){ c_kt += rdcycle()-c0m; i_kt += rdinstret()-i0m; }

    for(size_t i = 0; i < N; i += Br){
        size_t br = (N - i < Br) ? (N - i) : Br;
        for(size_t a = 0; a < br; ++a){ max_q[a] = -3.402823466e+38f; sum_q[a] = 0.0f; alpha_row[a] = 1.0f; }
        float *O_i = O + i * d;

        for(size_t j = 0; j < N; j += Bc){
            size_t bc = (N - j < Bc) ? (N - j) : Bc;

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // Kt_j 拷贝
            for(size_t k = 0; k < d; ++k)
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(K_T + k * N + j + b, vl);
                    __riscv_vse32_v_f32m4(Kt_j + k * bc + b, v, vl);
                }
            if(g_prof){ c_cp += rdcycle()-c0m; i_cp += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // matmul
            matmul_rvv(Q + i * d, Kt_j, S_ij, br, d, bc);
            if(g_prof){ c_mm += rdcycle()-c0m; i_mm += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // S *= scale
            for(size_t a = 0; a < br * bc; a += vl){
                vl = __riscv_vsetvl_e32m4(br * bc - a);
                vfloat32m4_t s = __riscv_vle32_v_f32m4(S_ij + a, vl);
                s = __riscv_vfmul_vf_f32m4(s, scale, vl);
                __riscv_vse32_v_f32m4(S_ij + a, s, vl);
            }
            if(g_prof){ c_sc += rdcycle()-c0m; i_sc += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // online softmax
            for(size_t a = 0; a < br; ++a)
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t p = online_softmax_rvv(S_ij + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);
                    __riscv_vse32_v_f32m4(P_ij + a * bc + b, p, vl);
                }
            if(g_prof){ c_sm += rdcycle()-c0m; i_sm += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // O 修正 + P@V
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
            if(g_prof){ c_oa += rdcycle()-c0m; i_oa += rdinstret()-i0m; }
        }
        if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }       // 归一化
        for(size_t a = 0; a < br; ++a)
            for(size_t p = 0; p < d; p += vl){
                vl = __riscv_vsetvl_e32m4(d - p);
                vfloat32m4_t o = __riscv_vle32_v_f32m4(O_i + a * d + p, vl);
                float inv = 1.0f / sum_q[a];
                o = __riscv_vfmul_vf_f32m4(o, inv, vl);
                __riscv_vse32_v_f32m4(O_i + a * d + p, o, vl);
            }
        if(g_prof){ c_nm += rdcycle()-c0m; i_nm += rdinstret()-i0m; }
    }
}

// ---------------- main：bf16 存储层 + 单次 fp32 内核计时 ----------------
static uint16_t Qh[ND], Kh[ND], Vh[ND], Oh[ND];     // bf16 存储格式
static float    Qf[ND], Kf[ND], Vf[ND], Of[ND];     // fp32 工作区（由 bf16 解码而来）

int main(){
    const size_t N = BENCH_N, d = BENCH_D;

    // 生成 fp32 原始数据，编码成 bf16 存入 Qh/Kh/Vh（“内存里只放 bf16”）
    for(size_t i = 0; i < ND; ++i){
        Qf[i] = (float)(i % 10) * 0.1f;
        Kf[i] = (float)(i % 7) * 0.1f;
        Vf[i] = (float)(i % 5) * 0.1f;
    }
    uint64_t c0e = rdcycle(), i0e = rdinstret();
    f32buf_to_bf16(Qf, Qh, ND); f32buf_to_bf16(Kf, Kh, ND); f32buf_to_bf16(Vf, Vh, ND);
    uint64_t c_enc0 = rdcycle() - c0e, i_enc0 = rdinstret() - i0e;
    printf("data_ok\n");

    // 单次计时：bf16 → fp32 解码（计格式转换代价）→ fp32 内核 → O 编码回 bf16
    uint64_t c0 = rdcycle(), i0 = rdinstret();
    bf16_to_f32buf(Qh, Qf, ND); bf16_to_f32buf(Kh, Kf, ND); bf16_to_f32buf(Vh, Vf, ND);  // 解码（源=bf16，目标=fp32）
    uint64_t c_dec = rdcycle() - c0, i_dec = rdinstret() - i0;

    const int R = 1;
    uint64_t cf = 0, in_f = 0;
    g_prof = 1;
    for(int r = 0; r < R; ++r){
        uint64_t c1 = rdcycle(), i1 = rdinstret();
        flash_attention_rvv(Qf, Kf, Vf, Of, N, d);          // 全 fp32 计算
        cf += rdcycle() - c1;  in_f += rdinstret() - i1;
    }
    g_prof = 0;

    c0 = rdcycle(); i0 = rdinstret();
    f32buf_to_bf16(Of, Oh, ND);                              // 输出存回 bf16
    uint64_t c_enc = rdcycle() - c0, i_enc = rdinstret() - i0;
    printf("flash_ok\n");

    printf("=== Saturn REFV256D128: RVV flash bf16存储/fp32计算 (single) ===\n");
    printf("N=%u d=%u reps=%d (avg)\n", (unsigned)N, (unsigned)d, R);
    printf("RVV   cycles = %lu  instret = %lu\n", (unsigned long)(cf/R), (unsigned long)(in_f/R));
    printf("--- 格式转换(标量, 一次整阵) ---\n");
    printf("编码QKV(f32→bf16) cyc=%lu ins=%lu\n", (unsigned long)c_enc0, (unsigned long)i_enc0);
    printf("解码bf16→f32      cyc=%lu ins=%lu\n", (unsigned long)c_dec,   (unsigned long)i_dec);
    printf("编码O(f32→bf16)   cyc=%lu ins=%lu\n", (unsigned long)c_enc,   (unsigned long)i_enc);
    printf("--- RVV 内核分阶段 (avg/R) ---\n");
    printf("K转置   cyc=%lu ins=%lu\n", (unsigned long)(c_kt/R), (unsigned long)(i_kt/R));
    printf("Kt_j    cyc=%lu ins=%lu\n", (unsigned long)(c_cp/R), (unsigned long)(i_cp/R));
    printf("matmul  cyc=%lu ins=%lu\n", (unsigned long)(c_mm/R), (unsigned long)(i_mm/R));
    printf("S*scale cyc=%lu ins=%lu\n", (unsigned long)(c_sc/R), (unsigned long)(i_sc/R));
    printf("softmax cyc=%lu ins=%lu\n", (unsigned long)(c_sm/R), (unsigned long)(i_sm/R));
    printf("  其中exp cyc=%lu ins=%lu\n", (unsigned long)(c_exp/R), (unsigned long)(i_exp/R));
    printf("  alpha标量exp cyc=%lu ins=%lu\n", (unsigned long)(c_ep/R), (unsigned long)(i_ep/R));
    printf("O累加   cyc=%lu ins=%lu\n", (unsigned long)(c_oa/R), (unsigned long)(i_oa/R));
    printf("归一化  cyc=%lu ins=%lu\n", (unsigned long)(c_nm/R), (unsigned long)(i_nm/R));
    printf("Oh[0..3] raw= %04x %04x %04x %04x  (bf16 存储样例)\n",
           Oh[0], Oh[1], Oh[2], Oh[3]);
    return 0;
}
