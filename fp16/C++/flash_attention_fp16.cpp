//编译：riscv64-unknown-linux-gnu-g++ -march=rv64gcv_zvfh -O2 -c flash_attention_fp16.cpp -o flash_attention_fp16.o

#include<riscv_vector.h>
#include<iostream>
#include<vector>            //动态数组容器
#include<cmath>             //数学函数库
#include<cfloat>            //获取浮点数类型特性
#include<algorithm>         //std::min


//矩阵乘法（4 行寄存器分块：B 每行只载一次，4 个 A 标量行复用）
//与 flash_attention.cpp 的 matmul_rvv 逐行对应，仅把 e32/m4/f32m4 换成 e16/m4/f16m4
static void matmul_rvv_f16(const _Float16 *A, const _Float16 *B, _Float16 *C, size_t M, size_t K, size_t N){
    const size_t TILE = 4;                      // 一次 4 行
    size_t vl = __riscv_vsetvl_e16m4(N);        // N=bc≤64 恒满宽
    size_t i = 0;
    //完整 4 行块：GCC 禁止 RVV 类型作数组元素，故用 4 个命名累加器（即 acc[0..3]）
    for(; i + TILE <= M; i += TILE){
        vfloat16m4_t acc0 = __riscv_vfmv_v_f_f16m4(0.0f, vl);
        vfloat16m4_t acc1 = __riscv_vfmv_v_f_f16m4(0.0f, vl);
        vfloat16m4_t acc2 = __riscv_vfmv_v_f_f16m4(0.0f, vl);
        vfloat16m4_t acc3 = __riscv_vfmv_v_f_f16m4(0.0f, vl);
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
        vfloat16m4_t acc = __riscv_vfmv_v_f_f16m4(0.0f, vl);
        for(size_t k = 0; k < K; ++k){
            const _Float16 *b_row = B + k * N;
            vfloat16m4_t b_vec = __riscv_vle16_v_f16m4(b_row, vl);
            acc = __riscv_vfmacc_vf_f16m4(acc, A[i * K + k], b_vec, vl);
        }
        __riscv_vse16_v_f16m4(C + i * N, acc, vl);
    }
}



//exp近似，帕德近似（x=-10失效；实测 |x|≲2 内才准，x≈-3.8 起变负）：
//  e^x = （1 + x/2 + 9*x^2/56 + 5*x^3/168 + x^4/1680）/ （1 - x/2 + 9*x^2/56 - 5*x^3/168 + x^4/1680）
//与 flash_attention.cpp 的 rvv_exp 同一套系数；fp16 下多项式各项自身的舍入会再引入约 1e-3 的相对噪声，
//但远小于近似式本身的误差（x=-2 处已偏 6%），所以精度瓶颈在近似式的有效区间，不在这里。
inline vfloat16m4_t rvv_exp_f16(vfloat16m4_t x, size_t vl){
    //帕德系数
    const _Float16 c4 = (_Float16)(1.0f / 1680.0f);    // 1/1680
    const _Float16 c3 = (_Float16)(5.0f / 168.0f);     // 5/168
    const _Float16 c2 = (_Float16)(9.0f / 56.0f);      // 9/56
    const _Float16 c1 = (_Float16)(1.0f / 2.0f);       //1/2
    const _Float16 c0 = (_Float16)(1.0f / 1.0f);       //1/1


    //分子
    vfloat16m4_t r_mol = __riscv_vfmul_vf_f16m4(x, c4, vl);
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c3, vl);
    r_mol = __riscv_vfmul_vv_f16m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c2, vl);
    r_mol = __riscv_vfmul_vv_f16m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c1, vl);
    r_mol = __riscv_vfmul_vv_f16m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f16m4(r_mol, c0, vl);

    //分母
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

    return r;
}


//online_softmax算法
//online_input部分矩阵；online_max运行最大值；online_sum运行和；alpha修正因子
//三个行级运行量保持 float：它们跨 tile 累积，放 fp16 会随 tile 数漂移。
//返回值是 fp16 的未归一化概率 P，直接喂给后面的 P·V（中间不再出现 fp32 向量）。
inline vfloat16m4_t online_softmax_f16(const _Float16 *online_input, float &online_max, float &online_sum, float &alpha, size_t vl){
    //safe softmax
    //寻找最大值（max 只做比较，fp16 精确，与 fp32 路径给出同一个结果）
    const float NEG_INF = -INFINITY;
    vfloat16m4_t online_vec = __riscv_vle16_v_f16m4(online_input, vl);

    //求本块最大值
    vfloat16m1_t acc  = __riscv_vfmv_s_f_f16m1((_Float16)NEG_INF, vl);
    vfloat16m1_t vmax = __riscv_vfredmax_vs_f16m4_f16m1(online_vec, acc, vl);
    float max_block   = (float)__riscv_vfmv_f_s_f16m1_f16(vmax);
    float max_new     = (online_max > max_block) ? online_max : max_block;

    //修正因子计算
    alpha = std::exp(online_max - max_new);//标量可以直接调用C++语言中的exp函数
    online_max = max_new;                  //新的最大值覆盖原最大值

    //safe softmax
    online_vec = __riscv_vfsub_vf_f16m4(online_vec, (_Float16)max_new, vl);
    vfloat16m4_t online_output = rvv_exp_f16(online_vec, vl);//online_output是未进行归一化的输出

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


//主函数
//实现K转置，矩阵分块，归一化
void flash_attention_fp16_rvv (const _Float16 *Q, const _Float16 *K, const _Float16 *V, _Float16 *O,size_t N, size_t d){
    const _Float16 scale = (_Float16)(1.0f / std::sqrt((float)d));
    const size_t Br = 64;//Q块大小（f16m4 一组 64 个元素，刚好填满向量组；fp32 版此处为 32）
    const size_t Bc = 64;//KV块大小
    size_t vl;

    //对O清零
    for(size_t io = 0; io < N * d; ++io)
        O[io] = (_Float16)0.0f;

    //对K进行转置
    std::vector<_Float16>K_T(d * N);
    for(size_t k = 0; k < d; ++k){
        for(size_t j = 0; j < N; j += vl){
        vl = __riscv_vsetvl_e16m4(N - j);
        vfloat16m4_t k_col = __riscv_vlse16_v_f16m4(K + j * d + k, sizeof(_Float16) * d, vl);
        __riscv_vse16_v_f16m4(K_T.data() + k * N + j, k_col, vl);
        }
    }


    //缓冲
    std::vector<_Float16>Kt_j(d * Bc);
    std::vector<_Float16>S_ij(Br * Bc);//S = Q * K^T
    std::vector<_Float16>P_ij(Br * Bc);//softmax概率


    //外层循环，对Q切块
    for(size_t i = 0; i < N; i += Br){
        size_t br = std::min(Br, N - i);
        std::vector<float>max_q(br, -INFINITY);
        std::vector<float>sum_q(br, 0.0f);
        std::vector<float>alpha_row(br, 1.0f);//每行的修正因子
        _Float16 *O_i = O + i * d;


        //内层循环，对KV进行切块
        for(size_t j = 0; j < N; j += Bc){
            size_t bc = std::min(Bc, N - j);

            //对K_T进行切块：逐行把 K_T[k][j..j+bc) 向量拷贝成连续 Kt_j[k][0..bc)
            for(size_t k = 0; k < d; ++k){
                for(size_t b = 0; b < bc; b += vl){
                    vl = __riscv_vsetvl_e16m4(bc - b);
                    vfloat16m4_t v = __riscv_vle16_v_f16m4(K_T.data() + k * N + j + b, vl);
                    __riscv_vse16_v_f16m4(Kt_j.data() + k * bc + b, v, vl);
                }
            }
                    matmul_rvv_f16(Q + i * d, Kt_j.data(), S_ij.data(), br, d, bc);//分块乘法得出矩阵S

                    //分块矩阵S向量化（fp16 乘 fp16 标量，8 个元素/周期）
                    for(size_t a = 0; a < br * bc; a += vl){
                        vl = __riscv_vsetvl_e16m4(br * bc - a);
                        vfloat16m4_t s = __riscv_vle16_v_f16m4(S_ij.data() + a, vl);
                        s = __riscv_vfmul_vf_f16m4(s, scale, vl);
                        __riscv_vse16_v_f16m4(S_ij.data() + a, s, vl);
                    }


                    //按行做online_softmax
                    for(size_t a = 0; a < br; ++a){
                         for (size_t b = 0; b < bc; b += vl) {
                            vl = __riscv_vsetvl_e16m4(bc - b);
                            vfloat16m4_t p = online_softmax_f16(S_ij.data() + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);
                            __riscv_vse16_v_f16m4(P_ij.data() + a * bc + b, p, vl);
                        }
                    }

                    //O_i修正和累加
                    for(size_t a = 0; a < br; ++a){
                        _Float16 *o_row = O_i + a * d;

                        //乘修正因子
                        for(size_t p = 0; p < d; p += vl){
                            vl = __riscv_vsetvl_e16m4(d - p);
                            vfloat16m4_t o = __riscv_vle16_v_f16m4(o_row + p, vl);
                            o = __riscv_vfmul_vf_f16m4(o, (_Float16)alpha_row[a], vl);
                            __riscv_vse16_v_f16m4(o_row + p, o, vl);
                        }

                        //累加
                        for(size_t b = 0; b <bc; ++b){
                            _Float16 pad = P_ij[a * bc + b];
                            const _Float16 *v_row = V + (j + b) * d;
                            for(size_t p = 0; p < d; p += vl){
                                vl =__riscv_vsetvl_e16m4(d - p);
                                vfloat16m4_t o = __riscv_vle16_v_f16m4(o_row + p, vl);
                                vfloat16m4_t v = __riscv_vle16_v_f16m4(v_row + p, vl);
                                o = __riscv_vfmacc_vf_f16m4(o, pad, v, vl);
                                __riscv_vse16_v_f16m4(o_row + p, o, vl);
                            }
                        }
                    }
        }
        //归一化
        for(size_t a =0; a < br; ++a){
            _Float16 *o_row =O_i + a * d;
            for(size_t p = 0; p < d; p += vl){
                vl = __riscv_vsetvl_e16m4(d - p);
                vfloat16m4_t o = __riscv_vle16_v_f16m4(o_row + p, vl);
                _Float16 inv = (_Float16)(1.0f / sum_q[a]);
                o = __riscv_vfmul_vf_f16m4(o, inv, vl);
                __riscv_vse16_v_f16m4(o_row + p, o, vl);
            }
        }
    }
}
