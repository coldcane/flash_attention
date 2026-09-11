//编译：riscv64-unknown-linux-gnu-g++ -march=rv64gcv -O2 -c flash_attention.cpp -o flash_attention.o

#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include<riscv_vector.h>
#include<vector>            //动态数组容器
#include<cmath>             //数学函数库
#include<cfloat>            //获取浮点数类型特性



//矩阵乘法
inline void matmul_rvv(const float *A, const float *B, float *C, size_t M, size_t K,size_t N){
    //初始化为零
    size_t vl;
    for(size_t i = 0; i < M; ++i){
        float *c_row = C + i * N;
        size_t j = 0;
        for(;j < N; j += vl){
            vl = __riscv_vsetvl_e32m4(N - j);//剩余元素和硬件上限取小
            vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.0f,vl);
            __riscv_vse32_v_f32m4(c_row + j, zero, vl);//将0写入C中的第i行
        }
    }


    //计算矩阵相乘
    for(size_t i = 0; i < M; ++i){
        float *c_row = C + i * N;
        for(size_t k = 0; k < K; ++k){
            float aik = A[i * K + k];
            const float *b_row = B + k * N;
            size_t j = 0;
            for(; j < N; j += vl){
                vl = __riscv_vsetvl_e32m4(N - j);
                vfloat32m4_t c_vec = __riscv_vle32_v_f32m4(c_row + j,vl); 
                vfloat32m4_t b_vec = __riscv_vle32_v_f32m4(b_row + j,vl);
                c_vec = __riscv_vfmacc_vf_f32m4(c_vec, aik, b_vec, vl);
                __riscv_vse32_v_f32m4(c_row + j, c_vec, vl); 
            }
        }
    }
}


//exp近似，帕德近似（x=-10失效）：e^x = （1 + x/2 + 9*x^2/56 + 5*x^3/168 + x^4/1680）/ （1 - x/2 + 9*x^2/56 - 5*x^3/168 + x^4/1680）
inline vfloat32m4_t rvv_exp(vfloat32m4_t x, size_t vl){
    //帕德系数
    const float c4 = 1.0f / 1680.0f;    // 1/1680
    const float c3 = 5.0f / 168.0f;     // 5/168
    const float c2 = 9.0f / 56.0f;      // 9/56
    const float c1 = 1.0f / 2.0f;       //1/2
    const float c0 = 1.0f / 1.0f;       //1/1


    //分子
    vfloat32m4_t r_mol = __riscv_vfmul_vf_f32m4(x, c4, vl);
    r_mol = __riscv_vfadd_vf_f32m4(r_mol, c3, vl);
    r_mol = __riscv_vfmul_vv_f32m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f32m4(r_mol, c2, vl);
    r_mol = __riscv_vfmul_vv_f32m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f32m4(r_mol, c1, vl);
    r_mol = __riscv_vfmul_vv_f32m4(r_mol, x, vl );
    r_mol = __riscv_vfadd_vf_f32m4(r_mol, c0, vl);

    //分母
    vfloat32m4_t r_den = __riscv_vfmul_vf_f32m4(x, c4, vl);
    r_den = __riscv_vfsub_vf_f32m4(r_den, c3, vl);
    r_den = __riscv_vfmul_vv_f32m4(r_den, x, vl );
    r_den = __riscv_vfadd_vf_f32m4(r_den, c2, vl);
    r_den = __riscv_vfmul_vv_f32m4(r_den, x, vl );
    r_den = __riscv_vfsub_vf_f32m4(r_den, c1, vl);
    r_den = __riscv_vfmul_vv_f32m4(r_den, x, vl );
    r_den = __riscv_vfadd_vf_f32m4(r_den, c0, vl);

    //相除
    vfloat32m4_t r = __riscv_vfdiv_vv_f32m4(r_mol, r_den, vl);

    return r;
}

//online_softmax算法
//online_input部分矩阵；online_max运行最大值；online_sum运行和；alpha修正因子
inline vfloat32m4_t online_softmax_rvv(const float *online_input, float &online_max, float &online_sum,float &alpha, size_t vl){
    //safe softmax
    //寻找最大值
    float max_val = -INFINITY;
    vfloat32m4_t online_vec = __riscv_vle32_v_f32m4(online_input, vl);

    //求本块最大值
    vfloat32m1_t acc  = __riscv_vfmv_s_f_f32m1(max_val,vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m4_f32m1(online_vec, acc, vl);
    float max_block   = __riscv_vfmv_f_s_f32m1_f32(vmax);
    float max_new     = (online_max > max_block) ? online_max : max_block;

    //修正因子计算
    alpha = std::exp(online_max - max_new);//标量可以直接调用C++语言中的exp函数
    online_max = max_new;                  //新的最大值覆盖原最大值

    //safe softmax
    online_vec = __riscv_vfsub_vf_f32m4(online_vec, max_new, vl);
    vfloat32m4_t online_output = rvv_exp(online_vec, vl);//online_output是未进行归一化的输出

    //求运行和，需要修正旧的运行和
    vfloat32m1_t zero_acc = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t vsum     = __riscv_vfredusum_vs_f32m4_f32m1(online_output, zero_acc, vl);
    online_sum = online_sum * alpha + __riscv_vfmv_f_s_f32m1_f32(vsum);


    return online_output;
}


//主函数
//实现K转置，矩阵分块，归一化
inline void flash_attention_rvv (const float *Q, const float *K, const float *V, float *O,size_t N, size_t d){
    const float scale = 1.0f /std::sqrt((float)d);
    const size_t Br = 32;//Q块大小
    const size_t Bc = 32;//KV块大小
    size_t vl;

    //对O清零
    for(size_t io = 0; io < N * d; ++io)
        O[io] = 0.0f;
    
    //对K进行转置
    std::vector<float>K_T(d * N);
    for(size_t k = 0; k < d; ++k){
        for(size_t j = 0; j < N; j += vl){
        vl = __riscv_vsetvl_e32m4(N - j);
        vfloat32m4_t k_col = __riscv_vlse32_v_f32m4(K + j * d + k, sizeof(float) * d, vl);
        __riscv_vse32_v_f32m4(K_T.data() + k * N + j, k_col, vl);
        }
    }


    //缓冲
    std::vector<float>Kt_j(d * Bc);
    std::vector<float>S_ij(Br * Bc);//S = Q * K^T
    std::vector<float>P_ij(Br * Bc);//softmax概率


    //外层循环，对Q切块
    for(size_t i = 0; i < N; i += Br){
        size_t br = std::min(Br, N - i);
        std::vector<float>max_q(br, -INFINITY);
        std::vector<float>sum_q(br, 0.0f);
        std::vector<float>alpha_row(br, 1.0f);//每行的修正因子
        float *O_i = O + i * d;
        

        //内层循环，对KV进行切块
        for(size_t j = 0; j < N; j += Bc){
            size_t bc = std::min(Bc, N - j);

            //对K_T进行切块
            for(size_t k = 0; k < d; ++k)
                for(size_t b = 0; b < bc; ++b)
                    Kt_j[k * bc + b] = K_T[k * N + j + b];
                    matmul_rvv(Q + i * d, Kt_j.data(), S_ij.data(), br, d, bc);//分块乘法得出矩阵S
                    
                    //分块矩阵S向量化
                    for(size_t a = 0; a < br * bc; a += vl){
                        vl = __riscv_vsetvl_e32m4(br * bc - a);
                        vfloat32m4_t s = __riscv_vle32_v_f32m4(S_ij.data() + a, vl);
                        s = __riscv_vfmul_vf_f32m4(s, scale, vl);
                        __riscv_vse32_v_f32m4(S_ij.data() + a, s, vl);
                    }


                    //按行做online_softmax
                    for(size_t a = 0; a < br; ++a){
                         for (size_t b = 0; b < bc; b += vl) {                 // ← 新增内层循环
                            vl = __riscv_vsetvl_e32m4(bc - b);                 // ← 每块算合法 vl（≤16）
                            vfloat32m4_t p = online_softmax_rvv(S_ij.data() + a * bc + b, max_q[a], sum_q[a], alpha_row[a], vl);         
                            __riscv_vse32_v_f32m4(P_ij.data() + a * bc + b, p, vl);
                        }
                    }

                    //O_i修正和累加
                    for(size_t a = 0; a < br; ++a){
                        float *o_row = O_i + a * d;

                        //乘修正因子
                        for(size_t p = 0; p < d; p += vl){
                            vl = __riscv_vsetvl_e32m4(d - p);
                            vfloat32m4_t o = __riscv_vle32_v_f32m4(o_row + p, vl);
                            o = __riscv_vfmul_vf_f32m4(o, alpha_row[a], vl);
                            __riscv_vse32_v_f32m4(o_row + p, o,vl);
                        }

                        //累加
                        for(size_t b = 0; b <bc; ++b){
                            float pad = P_ij[a * bc + b];
                            const float *v_row = V + (j + b) * d;
                            for(size_t p = 0; p < d; p += vl){
                                vl =__riscv_vsetvl_e32m4(d - p);
                                vfloat32m4_t o = __riscv_vle32_v_f32m4(o_row + p, vl);
                                vfloat32m4_t v = __riscv_vle32_v_f32m4(v_row + p, vl);
                                o = __riscv_vfmacc_vf_f32m4(o, pad, v, vl);
                                __riscv_vse32_v_f32m4(o_row + p, o, vl);
                            }
                        }
                    }
        }
        //归一化
        for(size_t a =0; a < br; ++a){
            float *o_row =O_i + a * d;
            for(size_t p = 0; p < d; p += vl){
                vl = __riscv_vsetvl_e32m4(d - p);
                vfloat32m4_t o = __riscv_vle32_v_f32m4(o_row + p, vl);
                o = __riscv_vfdiv_vf_f32m4(o, sum_q[a], vl);
                __riscv_vse32_v_f32m4(o_row + p, o, vl);
            } 
        }
    }
}

#endif