//编译：riscv64-unknown-linux-gnu-g++ -march=rv64gcv -O2 -static test_flash.cpp -o test_flash
//运行：spike --isa=rv64gcv pk test_flash

#include <iostream>   // std::cout
#include <vector>     // std::vector
#include <cmath>      // std::sin, std::cos, std::exp, std::fabs
#include <cfloat>     // INFINITY
#include <cstddef>    // size_t

#include "flash_attention.cpp"   // 直接把实现 include 进来

// 朴素标量版 attention：全量 S = QK^T，标准 softmax，P@V（无分块、无 online）
// 作为基准，用于验证 flash_attention_rvv 的正确性
void naive_attention(const float *Q, const float *K, const float *V,
                     float *O, size_t N, size_t d) {
    const float scale = 1.0f / std::sqrt((float)d);

    for (size_t i = 0; i < N; ++i) {              // 每个 query 行
        // ① 算 S[i][j] = Q_i·K_j * scale，同时找 max
        float m = -INFINITY;
        std::vector<float> s(N);
        for (size_t j = 0; j < N; ++j) {
            float dot = 0;
            for (size_t k = 0; k < d; ++k)
                dot += Q[i*d+k] * K[j*d+k];
            s[j] = dot * scale;
            if (s[j] > m) m = s[j];
        }
        // ② exp + 求和
        float sum = 0;
        std::vector<float> p(N);
        for (size_t j = 0; j < N; ++j) {
            p[j] = std::exp(s[j] - m);
            sum += p[j];
        }
        // ③ O[i] = Σ p[j]·V[j] / sum
        for (size_t k = 0; k < d; ++k) {
            float acc = 0;
            for (size_t j = 0; j < N; ++j)
                acc += p[j] * V[j*d+k];
            O[i*d+k] = acc / sum;
        }
    }
}

int main() {
    const size_t N = 64, d = 4;   // N=64 > Br=Bc=32，能真正触发分块

    // 用公式生成 Q/K/V（避免手写 256 个数）
    std::vector<float> Q(N*d), K(N*d), V(N*d);
    for (size_t i = 0; i < N*d; ++i) {
        Q[i] = std::sin((float)i * 0.5f);
        K[i] = std::cos((float)i * 0.3f);
        V[i] = std::sin((float)i * 0.7f);
    }
    std::vector<float> O_flash(N*d), O_naive(N*d);

    flash_attention_rvv(Q.data(), K.data(), V.data(), O_flash.data(), N, d);
    naive_attention  (Q.data(), K.data(), V.data(), O_naive.data(), N, d);

    // 对比：逐元素比较，允许浮点误差
    float max_diff = 0;
    for (size_t i = 0; i < N*d; ++i) {
        float diff = std::fabs(O_flash[i] - O_naive[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "max_diff = " << max_diff << std::endl;
    std::cout << (max_diff < 1e-2 ? ">>> PASSED" : ">>> FAILED") << std::endl;

    // 顺带打印 flash 版前 5 行，供人工查看
    std::cout << "O_flash (first 5 rows):\n";
    for (size_t i = 0; i < (N < 5 ? N : 5); ++i) {
        std::cout << "  O[" << i << "] = ";
        for (size_t j = 0; j < d; ++j)
            std::cout << O_flash[i*d+j] << " ";
        std::cout << std::endl;
    }

    // 打印朴素版前 5 行，方便和 flash 版直观对比
    std::cout << "O_naive (first 5 rows):\n";
    for (size_t i = 0; i < (N < 5 ? N : 5); ++i) {
        std::cout << "  O[" << i << "] = ";
        for (size_t j = 0; j < d; ++j)
            std::cout << O_naive[i*d+j] << " ";
        std::cout << std::endl;
    }
    return 0;
}