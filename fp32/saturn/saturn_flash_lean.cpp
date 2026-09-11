// saturn_flash_lean.cpp — Saturn 上 flash_attention_rvv (RVV) 计时版（瘦身）
// 从 saturn_flash_bench.cpp 派生：删去标量 naive_attention（含其预热/统计/正确性比对），
// 只保留 RVV flash 内核的预热 + R 次计时 + 分阶段 mcycle/minstret 统计。
// 说明：正确性比对(max_diff PASSED)依赖 naive 参照，随 naive 一并移除；
//        D=32 正确性已由 saturn_flash_bench.cpp 验证(PASSED, max_diff=0)。需要校验其它
//        尺寸/改动时，请用回 saturn_flash_bench.cpp（naive 会把墙钟拉长几倍）。
// 编译(conda 工具链): cd /home/coldcane/chipyard/tests && \
//   /home/coldcane/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-g++ \
//     -O2 -fno-common -fno-builtin-printf -march=rv64imafd_v -mabi=lp64d -mcmodel=medany \
//     -specs=/home/coldcane/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/lib/htif_nano.specs \
//     -static -T htif.ld saturn_flash_lean.cpp -o build/saturn_flash_lean.riscv
// 运行: cd /home/coldcane/chipyard/sims/verilator && \
//   ./simulator-chipyard.harness-REFV256D128RocketConfig \
//     /home/coldcane/chipyard/tests/build/saturn_flash_lean.riscv +verilator+seed+3
//
// ⚠️ 本版为「无打点」干净版：kernel 内不含任何 printf（原 wm2/wm_in 诊断打点已全部移除）。
//    后果：N=64 且 Br=Bc=32 时会**再次出现仿真停滞**。该停滞由「softmax/O累加 段长时间没有
//    host 往返」触发，只有靠行间 printf 维持节奏才能避开（2026-09-10 已定位并验证）。
//      · N≤63：不受影响，可正常跑完；
//      · 想在 N=64 下跑通：把 BENCH_BC 从 32 降到 16（列块变 partial width，绕过触发条件），
//        或改回带节奏打点的版本。
//    详见 recovery 手册 RESUME_saturn_flash_N64.md 与记忆库 n64-fullwidth-softmax-stall。

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define BENCH_N   64   
#define BENCH_D   256
#define BENCH_BR  16
#define BENCH_BC  16

// ---------------- M-mode 性能计数器 ----------------
static inline uint64_t rdcycle(void)   { uint64_t c; asm volatile("csrr %0, mcycle"   : "=r"(c)); return c; }
static inline uint64_t rdinstret(void) { uint64_t c; asm volatile("csrr %0, minstret" : "=r"(c)); return c; }

// 说明（2026-09-10）：原诊断版在这里有 wm2()（块级心跳打点）与 wm_in()（softmax/O累加 行首节奏
//   打点 + 自耗扣回）。两者都是为规避 N=64/D=256 的仿真停滞而存在——该停滞在长时间没有 host
//   往返的 softmax/O累加 段触发。**本版已全部移除**，kernel 内不含任何 printf。
//   ⚠️ 后果：N=64 且 Br=Bc=32 时会再次停滞（核心里没有 host 往返）。N≤63 不受影响。

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
            for(size_t a = 0; a < br; ++a){
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e32m4(bc - b);
                    vfloat32m4_t p = online_softmax_rvv(S_ij + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);
                    __riscv_vse32_v_f32m4(P_ij + a * bc + b, p, vl);
                }
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

// ---------------- main：单次计时（去预热，仅 RVV；诊断第二次调用是否卡死） ----------------
int main(){
    const size_t N = BENCH_N, d = BENCH_D;
    static float Q[BENCH_N * BENCH_D], K[BENCH_N * BENCH_D], V[BENCH_N * BENCH_D];
    static float O[BENCH_N * BENCH_D];

    for(size_t i = 0; i < N*d; ++i){
        Q[i] = (float)(i % 10) * 0.1f;
        K[i] = (float)(i % 7) * 0.1f;
        V[i] = (float)(i % 5) * 0.1f;
    }
    printf("data_ok\n");

    const int R = 1;
    uint64_t cf = 0, in_f = 0;
    g_prof = 1;                                        // 只跑一次：去预热，直接计时
    for(int r = 0; r < R; ++r){                        // 诊断：单次 g_prof=1 调用在 D=256 是否卡死
        uint64_t c0 = rdcycle(), i0 = rdinstret();
        flash_attention_rvv(Q, K, V, O, N, d);
        cf += rdcycle() - c0;  in_f += rdinstret() - i0;
    }
    g_prof = 0;
    printf("flash_ok\n");                              // 单次调用完成标记（在统计块前打出）

    printf("=== Saturn REFV256D128: RVV flash (lean, no naive) ===\n");
    printf("N=%u d=%u reps=%d (avg)\n", (unsigned)N, (unsigned)d, R);
    printf("RVV   cycles = %lu  instret = %lu\n",
           (unsigned long)(cf/R), (unsigned long)(in_f/R));

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
    printf("净总 ≈ %lu\n", (unsigned long)(cf/R));
    return 0;
}
