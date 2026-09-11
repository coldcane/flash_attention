// saturn_flash_fp16.cpp — 原生 fp16 向量计算的 flash attention（Saturn 裸机计时版）
//
// 本文件是桌面版 fp16/C++/flash_attention_fp16.cpp 的忠实移植，内核一一对应：
//   matmul_rvv_f16 / rvv_exp_f16 / online_softmax_f16 / flash_attention_fp16_rvv
//   分块（Br=Bc=64）、指令序列、注释位置全部照搬。只把"外壳"换成 Saturn 风格：
//     std::vector        → 静态数组
//     std::sqrt/std::exp → my_sqrtf / exp_pade（裸机不加 -lm，libm 会卡住）
//     无 main            → 加 main：数据入库 + 单次计时 + 分阶段 rdcycle 统计
//   正确性对照见桌面版（那边自带 fp32 参照）。
//
// ★ 与"fp16 存储 + fp32 计算"的本质区别：
//   S = Q·Kᵀ、S*scale、exp、P·V、O*=alpha、O/sum 全部走 SEW=16 的 fp16 向量指令，
//   一条吃 8 个元素（VLEN=256、DLEN=128），而 fp32 只有 4 个 —— 这才是"fp16 快一倍"的物理来源。
//   分块随之从 32 提到 64：VLEN=256 时 f16m4 一组 = 4×256/16 = 64 个元素，刚好填满向量组。
//   留在 fp32 的只有行级标量（每行只算一次，不影响元素吞吐）：
//     1. online_max / online_sum —— 跨 tile 累积的运行量，放 fp16 会随 tile 数漂移；
//     2. 行内求和用加宽归约（vfwredusum：加数位宽仍是 fp16，累加在 fp32 里做），理由同上。
//
// ⚠️ 前提：工具链需含 Zvfh（原生 fp16 向量算术）。只有 Zfh 的话下面所有 f16 向量指令都编不过。
//   已实测：chipyard 那套 conda 工具链（riscv64-unknown-elf-g++ 13.2.0）的 __riscv_zvfh = 0、
//   vfloat16m4_t 未声明，编不了本文件；需要换成带 Zvfh 的工具链。
//   编译：riscv64-unknown-elf-g++ -march=rv64gcv_zvfh -O2 -fno-common -fno-builtin-printf \
//           -mabi=lp64d -mcmodel=medany \
//           -specs=<toolchain>/riscv64-unknown-elf/lib/htif_nano.specs \
//           -static -T htif.ld saturn_flash_fp16.cpp -o build/saturn_flash_fp16.riscv
//   运行：与 saturn_flash_lean.md 相同流程，二进制换成 saturn_flash_fp16.riscv。
//
// ⚠️ 分块 Br=Bc=64，满宽需要 N≥64（BENCH_N<64 时 vsetvl 会自动钳短，结果仍对，只是半满）。
//   已知风险：N=64 且内核里没有 host 往返打点时会停滞，见 RESUME_saturn_flash_N64.md。
//   若长时间卡在 data_ok 之后不出结果，把 BENCH_N 降到 63 或更小即可绕开。

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define BENCH_N   64        // 满宽（f16m4 一组 64 个元素）；停机风险见文件头
#define BENCH_D   256
#define F16_BR    64        // Q 块大小：f16m4 一组 64 个元素，刚好填满向量组
#define F16_BC    64        // KV 块大小
#define ND        (BENCH_N * BENCH_D)

// ---------------- M-mode 性能计数器 ----------------
static inline uint64_t rdcycle(void)   { uint64_t c; asm volatile("csrr %0, mcycle"   : "=r"(c)); return c; }
static inline uint64_t rdinstret(void) { uint64_t c; asm volatile("csrr %0, minstret" : "=r"(c)); return c; }

// ---------------- 分阶段性能统计（g_prof=1 才统计） ----------------
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

static void prof_reset(void){
    c_kt=i_kt=c_cp=i_cp=c_mm=i_mm=c_sc=i_sc=0;
    c_sm=i_sm=c_oa=i_oa=c_nm=i_nm=c_exp=i_exp=c_ep=i_ep=0;
}

// ---------------- 标量数学（不用 libm，-lm 在 baremetal 下会卡住） ----------------
// 帕德近似 exp，系数与 rvv_exp_f16 一致
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
// 牛顿迭代 sqrt（对 d=256 这类整数足够精确）
static inline float my_sqrtf(float x){
    float r = x > 1.0f ? x : 1.0f;
    for(int i = 0; i < 20; ++i) r = 0.5f*(r + x/r);
    return r;
}

// ===========================================================================================
// 原生 fp16 内核 —— 与 flash_attention_fp16.cpp 逐行对应
// ===========================================================================================

// exp 近似，帕德近似，与桌面版 rvv_exp_f16 同一套系数：
//   e^x = （1 + x/2 + 9*x^2/56 + 5*x^3/168 + x^4/1680）/（1 - x/2 + 9*x^2/56 - 5*x^3/168 + x^4/1680）
// 全部在 fp16 里算（SEW=16，一条 8 个元素）；多项式各项自身的舍入引入约 1e-3 的相对噪声，
// 但远小于近似式本身的误差（x=-2 处已偏 6%），所以精度瓶颈在近似式的有效区间，不在这里。
inline vfloat16m4_t rvv_exp_f16(vfloat16m4_t x, size_t vl){
    uint64_t c0m = 0, i0m = 0;
    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }
    //帕德系数（显式转换 = 在这里从 float 舍入到 fp16）
    const _Float16 c4 = (_Float16)(1.0f / 1680.0f);    // 1/1680
    const _Float16 c3 = (_Float16)(5.0f / 168.0f);     // 5/168
    const _Float16 c2 = (_Float16)(9.0f / 56.0f);      // 9/56
    const _Float16 c1 = (_Float16)(1.0f / 2.0f);       // 1/2
    const _Float16 c0 = (_Float16)(1.0f / 1.0f);       // 1/1

    //分子
    vfloat16m4_t r_mol = __riscv_vfmul_vf_f16m4(x, c4, vl);
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c3, vl);
    r_mol = __riscv_vfmul_vv_f16m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c2, vl);
    r_mol = __riscv_vfmul_vv_f16m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c1, vl);
    r_mol = __riscv_vfmul_vv_f16m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c0, vl);

    //分母（奇次项取负）
    vfloat16m4_t r_den = __riscv_vfmul_vf_f16m4(x, c4, vl);
    r_den = __riscv_vfsub_vf_f16m4(r_den, c3, vl);
    r_den = __riscv_vfmul_vv_f16m4(r_den, x, vl );
    r_den = __riscv_vfadd_vf_f16m4(r_den, c2, vl);
    r_den = __riscv_vfmul_vv_f16m4(r_den, x, vl );
    r_den = __riscv_vfsub_vf_f16m4(r_den, c1, vl);
    r_den = __riscv_vfmul_vv_f16m4(r_den, x, vl );
    r_den = __riscv_vfadd_vf_f16m4(r_den, c0, vl);

    //相除
    vfloat16m4_t r = __riscv_vfdiv_vv_f16m4(r_mol, r_den, vl);
    if(g_prof){ c_exp += rdcycle()-c0m; i_exp += rdinstret()-i0m; }
    return r;
}

// 矩阵乘法（4 行寄存器分块：B 每行只载一次，4 个 A 标量行复用）
static void matmul_rvv_f16(const _Float16 *A, const _Float16 *B, _Float16 *C, size_t M, size_t K, size_t N){
    const size_t TILE = 4;                      // 一次 4 行
    size_t vl = __riscv_vsetvl_e16m4(N);        // N=bc≤64 恒满宽
    size_t i = 0;
    //完整 4 行块：GCC 禁止 RVV 类型作数组元素，故用 4 个命名累加器（即 acc[0..3]）
    for(; i + TILE <= M; i += TILE){
        vfloat16m4_t acc0 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl);
        vfloat16m4_t acc1 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl);
        vfloat16m4_t acc2 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl);
        vfloat16m4_t acc3 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl);
        for(size_t k = 0; k < K; ++k){          // 沿 K 维累加，结果留在寄存器
            const _Float16 *b_row = B + k * N;
            vfloat16m4_t b_vec = __riscv_vle16_v_f16m4(b_row, vl);
            acc0 = __riscv_vfmacc_vf_f16m4(acc0, A[(i + 0) * K + k], b_vec, vl);
            acc1 = __riscv_vfmacc_vf_f16m4(acc1, A[(i + 1) * K + k], b_vec, vl);
            acc2 = __riscv_vfmacc_vf_f16m4(acc2, A[(i + 2) * K + k], b_vec, vl);
            acc3 = __riscv_vfmacc_vf_f16m4(acc3, A[(i + 3) * K + k], b_vec, vl);
        }
        __riscv_vse16_v_f16m4(C + (i + 0) * N, acc0, vl);
        __riscv_vse16_v_f16m4(C + (i + 1) * N, acc1, vl);
        __riscv_vse16_v_f16m4(C + (i + 2) * N, acc2, vl);
        __riscv_vse16_v_f16m4(C + (i + 3) * N, acc3, vl);
    }
    //尾部 0~3 行：逐行点积
    for(; i < M; ++i){
        vfloat16m4_t acc = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vl);
        for(size_t k = 0; k < K; ++k){
            const _Float16 *b_row = B + k * N;
            vfloat16m4_t b_vec = __riscv_vle16_v_f16m4(b_row, vl);
            acc = __riscv_vfmacc_vf_f16m4(acc, A[i * K + k], b_vec, vl);
        }
        __riscv_vse16_v_f16m4(C + i * N, acc, vl);
    }
}

// online_softmax 算法
// online_input 部分矩阵；online_max 运行最大值；online_sum 运行和；alpha 修正因子
// 三个行级运行量保持 float：它们跨 tile 累积，放 fp16 会随 tile 数漂移。
// 返回值是 fp16 的未归一化概率 P，直接喂给后面的 P·V（中间不再出现 fp32 向量）。
inline vfloat16m4_t online_softmax_f16(const _Float16 *online_input, float &online_max,
                                       float &online_sum, float &alpha, size_t vl){
    //safe softmax
    //寻找最大值（max 只做比较，fp16 精确，与 fp32 路径给出同一个结果）
    const float NEG_INF = -3.402823466e+38f;
    vfloat16m4_t online_vec = __riscv_vle16_v_f16m4(online_input, vl);

    //求本块最大值
    vfloat16m1_t acc  = __riscv_vfmv_s_f_f16m1((_Float16)NEG_INF, vl);
    vfloat16m1_t vmax = __riscv_vfredmax_vs_f16m4_f16m1(online_vec, acc, vl);
    float max_block   = (float)__riscv_vfmv_f_s_f16m1_f16(vmax);
    float max_new     = (online_max > max_block) ? online_max : max_block;

    //修正因子计算。桌面版这里直接调 std::exp（libm），裸机不带 -lm，改用标量 exp_pade；
    //首块 online_max = -inf，exp_pade(-inf) 会算出 NaN，所以要显式短路成 0。
    alpha = (online_max > NEG_INF) ? exp_pade(online_max - max_new) : 0.0f;
    online_max = max_new;                  //新的最大值覆盖原最大值

    //safe softmax
    online_vec = __riscv_vfsub_vf_f16m4(online_vec, (_Float16)max_new, vl);
    vfloat16m4_t online_output = rvv_exp_f16(online_vec, vl);//online_output 是未进行归一化的输出

    //求运行和，需要修正旧的运行和
    //加宽归约求和：加数位宽仍是 fp16（不影响元素吞吐），但累加在 fp32 下做，
    //免得 64 项在 fp16 里累加的舍入误差随 tile 数累积成系统性漂移。
    //若工具链没有 vfwredusum，退回普通 fp16 归约即可：
    //  vfloat16m1_t vsum = __riscv_vfredusum_vs_f16m4_f16m1(online_output, __riscv_vfmv_s_f_f16m1((_Float16)0.0f, vl), vl);
    //  online_sum = online_sum * alpha + (float)__riscv_vfmv_f_s_f16m1_f16(vsum);
    vfloat32m1_t zero_acc = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t vsum     = __riscv_vfwredusum_vs_f16m4_f32m1(online_output, zero_acc, vl);
    online_sum = online_sum * alpha + __riscv_vfmv_f_s_f32m1_f32(vsum);

    return online_output;
}

// ---------------- flash_attention_fp16_rvv（静态缓冲，算法与桌面版一致） ----------------
// 缓冲尺寸全部取自内核实际使用的 F16_BR/F16_BC，保证不会写越界
static _Float16 K_Th[BENCH_D * BENCH_N], Kt_jh[BENCH_D * F16_BC];
static _Float16 S_ijh[F16_BR * F16_BC], P_ijh[F16_BR * F16_BC];
static float max_q[F16_BR], sum_q[F16_BR], alpha_row[F16_BR];

void flash_attention_fp16_rvv(const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                              _Float16 *O, size_t N, size_t d){
    const _Float16 scale = (_Float16)(1.0f / my_sqrtf((float)d));
    const size_t Br = F16_BR;   //Q块大小（f16m4 一组 64 个元素，刚好填满向量组；fp32 版此处为 32）
    const size_t Bc = F16_BC;   //KV块大小
    size_t vl;
    uint64_t c0m = 0, i0m = 0;
    const _Float16 zero = (_Float16)0.0f;

    //对O清零（标量，与参考版及 fp32/bf16 的 Saturn 版一致；
    //  这段在计时区内但没有任何分阶段计数器覆盖，改成向量化会让"净总"凭空变好，
    //  与 saturn_flash_lean.cpp 的横比就不可信了）
    for(size_t io = 0; io < N * d; ++io) O[io] = zero;

    if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }     // ── K 转置 ──
    for(size_t k = 0; k < d; ++k)
        for(size_t j = 0; j < N; j += vl){
            vl = __riscv_vsetvl_e16m4(N - j);
            vfloat16m4_t k_col = __riscv_vlse16_v_f16m4(K + j * d + k, (ptrdiff_t)(sizeof(_Float16) * d), vl);
            __riscv_vse16_v_f16m4(K_Th + k * N + j, k_col, vl);
        }
    if(g_prof){ c_kt += rdcycle()-c0m; i_kt += rdinstret()-i0m; }

    for(size_t i = 0; i < N; i += Br){                    //外层循环，对Q切块
        size_t br = (N - i < Br) ? (N - i) : Br;
        for(size_t a = 0; a < br; ++a){ max_q[a] = -3.402823466e+38f; sum_q[a] = 0.0f; alpha_row[a] = 1.0f; }
        _Float16 *O_i = O + i * d;

        for(size_t j = 0; j < N; j += Bc){                //内层循环，对KV进行切块
            size_t bc = (N - j < Bc) ? (N - j) : Bc;

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── Kt_j 拷贝 ──
            //对K_T进行切块：逐行把 K_T[k][j..j+bc) 向量拷贝成连续 Kt_j[k][0..bc)
            for(size_t k = 0; k < d; ++k)
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e16m4(bc - b);
                    vfloat16m4_t v = __riscv_vle16_v_f16m4(K_Th + k * N + j + b, vl);
                    __riscv_vse16_v_f16m4(Kt_jh + k * bc + b, v, vl);
                }
            if(g_prof){ c_cp += rdcycle()-c0m; i_cp += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── matmul ──
            matmul_rvv_f16(Q + i * d, Kt_jh, S_ijh, br, d, bc);//分块乘法得出矩阵S
            if(g_prof){ c_mm += rdcycle()-c0m; i_mm += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── S *= scale ──
            for(size_t a = 0; a < br * bc; a += vl){          //分块矩阵S向量化（fp16 乘 fp16 标量）
                vl = __riscv_vsetvl_e16m4(br * bc - a);
                vfloat16m4_t s = __riscv_vle16_v_f16m4(S_ijh + a, vl);
                s = __riscv_vfmul_vf_f16m4(s, scale, vl);
                __riscv_vse16_v_f16m4(S_ijh + a, s, vl);
            }
            if(g_prof){ c_sc += rdcycle()-c0m; i_sc += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── online softmax（含 exp）──
            for(size_t a = 0; a < br; ++a){                   //按行做 online_softmax
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e16m4(bc - b);
                    vfloat16m4_t p = online_softmax_f16(S_ijh + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);
                    __riscv_vse16_v_f16m4(P_ijh + a * bc + b, p, vl);
                }
            }
            if(g_prof){ c_sm += rdcycle()-c0m; i_sm += rdinstret()-i0m; }

            if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }   // ── O 修正 + P@V 累加 ──
            for(size_t a = 0; a < br; ++a){
                _Float16 *o_row = O_i + a * d;
                const _Float16 al = (_Float16)alpha_row[a];

                //乘修正因子
                for(size_t p = 0; p < d; p += vl){
                    vl = __riscv_vsetvl_e16m4(d - p);
                    vfloat16m4_t o = __riscv_vle16_v_f16m4(o_row + p, vl);
                    o = __riscv_vfmul_vf_f16m4(o, al, vl);
                    __riscv_vse16_v_f16m4(o_row + p, o, vl);
                }

                //累加
                for(size_t b = 0; b < bc; ++b){
                    _Float16 pad = P_ijh[a * bc + b];
                    const _Float16 *v_row = V + (j + b) * d;
                    for(size_t p = 0; p < d; p += vl){
                        vl = __riscv_vsetvl_e16m4(d - p);
                        vfloat16m4_t o = __riscv_vle16_v_f16m4(o_row + p, vl);
                        vfloat16m4_t v = __riscv_vle16_v_f16m4(v_row + p, vl);
                        o = __riscv_vfmacc_vf_f16m4(o, pad, v, vl);
                        __riscv_vse16_v_f16m4(o_row + p, o, vl);
                    }
                }
            }
            if(g_prof){ c_oa += rdcycle()-c0m; i_oa += rdinstret()-i0m; }
        }

        if(g_prof){ c0m = rdcycle(); i0m = rdinstret(); }       // ── 归一化 ──
        for(size_t a = 0; a < br; ++a){
            const _Float16 inv = (_Float16)(1.0f / sum_q[a]);
            for(size_t p = 0; p < d; p += vl){
                vl = __riscv_vsetvl_e16m4(d - p);
                vfloat16m4_t o = __riscv_vle16_v_f16m4(O_i + a * d + p, vl);
                o = __riscv_vfmul_vf_f16m4(o, inv, vl);
                __riscv_vse16_v_f16m4(O_i + a * d + p, o, vl);
            }
        }
        if(g_prof){ c_nm += rdcycle()-c0m; i_nm += rdinstret()-i0m; }
    }
}

// ---------------- main：fp16 数据直接入库 → 单次计时原生 fp16 内核 → 分阶段报告 ----------------
static _Float16 Qh[ND], Kh[ND], Vh[ND], Oh[ND];      // fp16 存储，也是内核的直接输入/输出

static void print_phases(void){
    printf("K转置   cyc=%lu ins=%lu\n", (unsigned long)c_kt, (unsigned long)i_kt);
    printf("Kt_j    cyc=%lu ins=%lu\n", (unsigned long)c_cp, (unsigned long)i_cp);
    printf("matmul  cyc=%lu ins=%lu\n", (unsigned long)c_mm, (unsigned long)i_mm);
    printf("S*scale cyc=%lu ins=%lu\n", (unsigned long)c_sc, (unsigned long)i_sc);
    printf("softmax cyc=%lu ins=%lu\n", (unsigned long)c_sm, (unsigned long)i_sm);
    printf("  其中exp cyc=%lu ins=%lu\n", (unsigned long)c_exp, (unsigned long)i_exp);
    printf("  alpha标量exp cyc=%lu ins=%lu\n", (unsigned long)c_ep, (unsigned long)i_ep);
    printf("O累加   cyc=%lu ins=%lu\n", (unsigned long)c_oa, (unsigned long)i_oa);
    printf("归一化  cyc=%lu ins=%lu\n", (unsigned long)c_nm, (unsigned long)i_nm);
}

int main(){
    const size_t N = BENCH_N, d = BENCH_D;

    //生成数据并编码成 fp16 入库（内核直接吃 fp16，无 fp32 工作区）
    uint64_t c0e = rdcycle(), i0e = rdinstret();
    for(size_t i = 0; i < ND; ++i){
        Qh[i] = (_Float16)((float)(i % 10) * 0.1f);
        Kh[i] = (_Float16)((float)(i % 7) * 0.1f);
        Vh[i] = (_Float16)((float)(i % 5) * 0.1f);
    }
    uint64_t c_enc0 = rdcycle() - c0e, i_enc0 = rdinstret() - i0e;
    printf("data_ok\n");

    const int R = 1;
    uint64_t ch = 0, in_h = 0;
    g_prof = 1; prof_reset();
    for(int r = 0; r < R; ++r){
        uint64_t c1 = rdcycle(), i1 = rdinstret();
        flash_attention_fp16_rvv(Qh, Kh, Vh, Oh, N, d);      // 原生 fp16 计算（SEW=16）
        ch += rdcycle() - c1;  in_h += rdinstret() - i1;
    }
    g_prof = 0;
    printf("flash_ok\n");

    printf("=== Saturn REFV256D128: flash fp16 原生计算 (SEW=16, 一组 64 元素) ===\n");
    printf("N=%u d=%u reps=%d (avg)\n", (unsigned)N, (unsigned)d, R);
    printf("生成+编码fp16  cyc=%lu ins=%lu\n", (unsigned long)c_enc0, (unsigned long)i_enc0);
    printf("RVV   cycles = %lu  instret = %lu\n",
           (unsigned long)(ch/R), (unsigned long)(in_h/R));
    printf("--- RVV 内核分阶段 (avg/R) ---\n");
    print_phases();
    printf("净总 ≈ %lu\n", (unsigned long)(ch/R));

    uint16_t oh0, oh1, oh2, oh3;
    memcpy(&oh0, &Oh[0], 2); memcpy(&oh1, &Oh[1], 2);
    memcpy(&oh2, &Oh[2], 2); memcpy(&oh3, &Oh[3], 2);
    printf("Oh[0..3] raw= %04x %04x %04x %04x  (fp16 存储样例)\n", oh0, oh1, oh2, oh3);
    return 0;
}
