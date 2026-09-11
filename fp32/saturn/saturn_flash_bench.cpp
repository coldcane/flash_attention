// saturn_flash_bench.cpp — 在 Saturn 上对比 flash_attention_rvv (RVV) vs naive_attention (标量)
// 测量 mcycle(周期→时间) 和 minstret(退役指令→资源量)，各跑 R 次取平均。
// 算法与 learning_project/flash_attention/ 下的版本一致，仅把 std::vector 换成静态数组以适配 baremetal。
// 编译: riscv64-unknown-elf-g++ -march=rv64imafd_v -mabi=lp64d -mcmodel=medany \
//          -O2 -static -specs=htif.specs -T htif.ld saturn_flash_bench.cpp \
//          -o build/saturn_flash_bench.riscv -lm
// 运行: ./simulator-chipyard.harness-REFV256D128RocketConfig \
//          /home/coldcane/chipyard/tests/build/saturn_flash_bench.riscv

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define BENCH_N   32
#define BENCH_D   256
#define BENCH_BR  32
#define BENCH_BC  32

// ---------------- M-mode 性能计数器 ----------------
static inline uint64_t rdcycle(void)   { uint64_t c; asm volatile("csrr %0, mcycle"   : "=r"(c)); return c; }
static inline uint64_t rdinstret(void) { uint64_t c; asm volatile("csrr %0, minstret" : "=r"(c)); return c; }

// ---------------- 分阶段性能统计（测各算子 cycles/instret，g_prof=1 才统计） ----------------
static int g_prof = 0;                  // 1=统计 0=不统计（预热）
static uint64_t c_kt, i_kt;             // K 转置
static uint64_t c_cp, i_cp;             // Kt_j 拷贝
static uint64_t c_mm, i_mm;             // matmul
static uint64_t c_sc, i_sc;             // S *= scale
static uint64_t c_sm, i_sm;             // online softmax（含 exp）
static uint64_t c_oa, i_oa;             // O 修正 + P@V 累加
static uint64_t c_nm, i_nm;             // 归一化
static uint64_t c_exp, i_exp;           // 纯 rvv_exp（softmax 的子集）
static uint64_t c_ep, i_ep;             // 标量 exp_pade（alpha 修正用）
static uint64_t c_ns, i_ns;             // naive S 计算
static uint64_t c_nsm, i_nsm;           // naive softmax（exp）
static uint64_t c_no, i_no;             // naive O 累加

// ---------------- 标量数学函数（不用 libm，-lm 在 baremetal 下会卡住） ----------------
// 帕德近似 exp，系数与 rvv_exp 一致
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
// 牛顿迭代 sqrt（对 d=4 这类小整数足够精确）
static inline float my_sqrtf(float x){
    float r = x > 1.0f ? x : 1.0f;
    for(int i = 0; i < 20; ++i) r = 0.5f*(r + x/r);
    return r;
}

// ---------------- RVV exp：(4,4) 帕德，纯浮点，无 float↔int 转换（回退版，用于二分定位） ----------------
static inline vfloat32m4_t rvv_exp(vfloat32m4_t x, size_t vl){
    uint64_t c0m = 0, i0m = 0;
    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }
    // (4,4) 帕德系数，与下方标量 exp_pade 完全一致
    const float c4 = 1.0f/1680.0f, c3 = 5.0f/168.0f,
                c2 = 9.0f/56.0f,   c1 = 1.0f/2.0f, c0 = 1.0f;
    const float d1 = -c1, d3 = -c3;              // 分母的负系数
    vfloat32m4_t v;
    // mol = c0 + x·(c1 + x·(c2 + x·(c3 + x·c4)))
    v = __riscv_vfmul_vf_f32m4(x, c4, vl);
    v = __riscv_vfadd_vf_f32m4(v, c3, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    v = __riscv_vfadd_vf_f32m4(v, c2, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    v = __riscv_vfadd_vf_f32m4(v, c1, vl);
    v = __riscv_vfmul_vv_f32m4(v, x, vl);
    vfloat32m4_t mol = __riscv_vfadd_vf_f32m4(v, c0, vl);
    // den = c0 + x·(d1 + x·(c2 + x·(d3 + x·c4)))
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

// ---------------- matmul RVV（4 行寄存器分块，与 flash_attention.cpp 一致） ----------------
static void matmul_rvv(const float *A, const float *B, float *C, size_t M, size_t K, size_t N){
    const size_t TILE = 4;                      // 一次 4 行
    size_t vl = __riscv_vsetvl_e32m4(N);        // N=bc≤32 恒满宽
    size_t i = 0;
    //完整 4 行块：GCC 禁止 RVV 类型作数组元素，故用 4 个命名累加器（即 acc[0..3]）
    for(; i + TILE <= M; i += TILE){
        vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for(size_t k = 0; k < K; ++k){          // 沿 K 维累加，结果留在寄存器
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
    //尾部 0~3 行：逐行点积
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

// ---------------- online softmax（同源，std::vector→static） ----------------
static vfloat32m4_t online_softmax_rvv(const float *online_input, float &online_max,
                                       float &online_sum, float &alpha, size_t vl){
    const float NEG_INF = -3.402823466e+38f;
    vfloat32m4_t online_vec = __riscv_vle32_v_f32m4(online_input, vl);
    vfloat32m1_t acc  = __riscv_vfmv_s_f_f32m1(NEG_INF, vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m4_f32m1(online_vec, acc, vl);
    float max_block = __riscv_vfmv_f_s_f32m1_f32(vmax);
    float max_new   = (online_max > max_block) ? online_max : max_block;
    alpha = (online_max > NEG_INF) ? exp_pade(online_max - max_new) : 0.0f;   // 标量修正因子（首块 online_max=-inf → alpha=0）
    online_max = max_new;
    online_vec = __riscv_vfsub_vf_f32m4(online_vec, max_new, vl);
    vfloat32m4_t online_output = rvv_exp(online_vec, vl);
    vfloat32m1_t zero_acc = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(online_output, zero_acc, vl);
    online_sum = online_sum * alpha + __riscv_vfmv_f_s_f32m1_f32(vsum);
    return online_output;
}

// ---------------- flash_attention_rvv（静态缓冲，算法与 flash_attention.cpp 一致；N=BENCH_N,d=BENCH_D） ----------------
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

    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }     // ── K 转置 ──
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

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── Kt_j 拷贝（向量，与 flash_attention.cpp 一致）──
            for(size_t k = 0; k < d; ++k){
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(K_T + k * N + j + b, vl);
                    __riscv_vse32_v_f32m4(Kt_j + k * bc + b, v, vl);
                }
            }
            if(g_prof){ c_cp += rdcycle()-c0m; i_cp += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── matmul ──
            matmul_rvv(Q + i * d, Kt_j, S_ij, br, d, bc);
            if(g_prof){ c_mm += rdcycle()-c0m; i_mm += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── S *= scale ──
            for(size_t a = 0; a < br * bc; a += vl){
                vl = __riscv_vsetvl_e32m4(br * bc - a);
                vfloat32m4_t s = __riscv_vle32_v_f32m4(S_ij + a, vl);
                s = __riscv_vfmul_vf_f32m4(s, scale, vl);
                __riscv_vse32_v_f32m4(S_ij + a, s, vl);
            }
            if(g_prof){ c_sc += rdcycle()-c0m; i_sc += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── online softmax（含 exp）──
            for(size_t a = 0; a < br; ++a)
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t p = online_softmax_rvv(S_ij + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);
                    __riscv_vse32_v_f32m4(P_ij + a * bc + b, p, vl);
                }
            if(g_prof){ c_sm += rdcycle()-c0m; i_sm += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── O 修正 + P@V 累加 ──
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
        if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }       // ── 归一化 ──
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

// ---------------- naive 标量版（同源，std::vector→static） ----------------
static float s_naive[BENCH_N], p_naive[BENCH_N];
void naive_attention(const float *Q, const float *K, const float *V,
                     float *O, size_t N, size_t d){
    const float scale = 1.0f / my_sqrtf((float)d);
    uint64_t c0m = 0, i0m = 0;
    for(size_t i = 0; i < N; ++i){
        float m = -3.402823466e+38f;

        if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }       // ── S 计算（dot+scale+max）──
        for(size_t j = 0; j < N; ++j){
            float dot = 0;
            for(size_t k = 0; k < d; ++k)
                dot += Q[i*d+k] * K[j*d+k];
            s_naive[j] = dot * scale;
            if(s_naive[j] > m) m = s_naive[j];
        }
        if(g_prof){ c_ns += rdcycle()-c0m; i_ns += rdinstret()-i0m; }

        if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }       // ── softmax（exp_pade+sum）──
        float sum = 0;
        for(size_t j = 0; j < N; ++j){ p_naive[j] = exp_pade(s_naive[j] - m); sum += p_naive[j]; }
        if(g_prof){ c_nsm += rdcycle()-c0m; i_nsm += rdinstret()-i0m; }

        if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }       // ── O 累加 ──
        for(size_t k = 0; k < d; ++k){
            float acc = 0;
            for(size_t j = 0; j < N; ++j)
                acc += p_naive[j] * V[j*d+k];
            O[i*d+k] = acc / sum;
        }
        if(g_prof){ c_no += rdcycle()-c0m; i_no += rdinstret()-i0m; }
    }
}

// ---------------- main：预热 + R 次测平均 ----------------
int main(){
    const size_t N = BENCH_N, d = BENCH_D;
    static float Q[BENCH_N * BENCH_D], K[BENCH_N * BENCH_D], V[BENCH_N * BENCH_D];
    static float O_flash[BENCH_N * BENCH_D], O_naive[BENCH_N * BENCH_D];

    for(size_t i = 0; i < N*d; ++i){
        Q[i] = (float)(i % 10) * 0.1f;
        K[i] = (float)(i % 7) * 0.1f;
        V[i] = (float)(i % 5) * 0.1f;
    }
    printf("data_ok\n");

    g_prof = 0;                                        // 预热：不统计
    flash_attention_rvv(Q, K, V, O_flash, N, d);   // 预热
    printf("flash_ok\n");
    naive_attention(Q, K, V, O_naive, N, d);
    printf("naive_ok\n");

    const int R = 1;
    uint64_t cf = 0, in_f = 0, cn = 0, in_n = 0;
    g_prof = 1;                                        // 正式循环：开始统计
    for(int r = 0; r < R; ++r){
        uint64_t c0 = rdcycle(), i0 = rdinstret();
        flash_attention_rvv(Q, K, V, O_flash, N, d);
        cf += rdcycle() - c0;  in_f += rdinstret() - i0;

        c0 = rdcycle(); i0 = rdinstret();
        naive_attention(Q, K, V, O_naive, N, d);
        cn += rdcycle() - c0;  in_n += rdinstret() - i0;
    }
    g_prof = 0;

    float max_diff = 0;
    for(size_t i = 0; i < N*d; ++i){
        float df = O_flash[i] - O_naive[i]; if(df < 0) df = -df; if(df > max_diff) max_diff = df;
    }
    printf("=== Saturn REFV256D128: RVV vs naive ===\n");
    printf("N=%u d=%u reps=%d (avg)\n", (unsigned)N, (unsigned)d, R);
    long md = (long)(max_diff * 1000000.0f);   // float 字面量，别用 double（硬件无 D 扩展）
    printf("max_diff = %ld (x1e-6)  [%s]\n", md, (max_diff < 1e-2f) ? "PASSED" : "FAILED");
    printf("RVV   cycles = %lu  instret = %lu\n",
           (unsigned long)(cf/R), (unsigned long)(in_f/R));
    printf("naive cycles = %lu  instret = %lu\n",
           (unsigned long)(cn/R), (unsigned long)(in_n/R));
    printf("cycle speedup x100 = %ld   instret ratio x100 = %ld\n",
           (long)((cn * 100) / cf),            // 整数比，避免 double 除法
           (long)((in_n * 100) / in_f));

    printf("--- RVV 分阶段 (avg/R) ---\n");
    printf("K转置   cyc=%lu ins=%lu\n", (unsigned long)(c_kt/R), (unsigned long)(i_kt/R));
    printf("Kt_j    cyc=%lu ins=%lu\n", (unsigned long)(c_cp/R), (unsigned long)(i_cp/R));
    printf("matmul  cyc=%lu ins=%lu\n", (unsigned long)(c_mm/R), (unsigned long)(i_mm/R));
    printf("S*scale cyc=%lu ins=%lu\n", (unsigned long)(c_sc/R), (unsigned long)(i_sc/R));
    printf("softmax cyc=%lu ins=%lu\n", (unsigned long)(c_sm/R), (unsigned long)(i_sm/R));
    printf("  其中exp cyc=%lu ins=%lu\n", (unsigned long)(c_exp/R), (unsigned long)(i_exp/R));
    printf("  alpha标量exp cyc=%lu ins=%lu\n", (unsigned long)(c_ep/R), (unsigned long)(i_ep/R));
    printf("O累加   cyc=%lu ins=%lu\n", (unsigned long)(c_oa/R), (unsigned long)(i_oa/R));
    printf("归一化  cyc=%lu ins=%lu\n", (unsigned long)(c_nm/R), (unsigned long)(i_nm/R));
    printf("--- naive 分阶段 (avg/R) ---\n");
    printf("S计算   cyc=%lu ins=%lu\n", (unsigned long)(c_ns/R), (unsigned long)(i_ns/R));
    printf("softmax cyc=%lu ins=%lu\n", (unsigned long)(c_nsm/R), (unsigned long)(i_nsm/R));
    printf("O累加   cyc=%lu ins=%lu\n", (unsigned long)(c_no/R), (unsigned long)(i_no/R));
    return 0;
}
