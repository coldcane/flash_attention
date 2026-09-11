# Saturn lean flash — 状态备忘（N=64 停滞：已解决）

> 最后更新：**2026-09-10**。本文档独立于 Claude 记忆，随时可读。
> 旧版标题是「N=64 竞态调查」，结论是"未解决" —— 那份内容已作废，勿再参考。

## 1. 结论

**N=64 / D=256 / Br=Bc=32 的停滞已定位并绕过**，kernel 能完整跑完并输出分阶段报告。

现象：跑到**块 (i=0, j=32)** 的 online softmax 就永久停滞 —— 进程 100% CPU 空转、不再产出任何输出、**永不退出**。

根因是**仿真层面时序敏感的停滞**，三条排除证据：
- 不是软件 bug：同一份代码 N=32/48/60/63 全部跑通。
- 不是"某条满宽向量指令死锁"：加打点后同样的指令流跑得过去。
- 不是"单纯太慢"：在 softmax/O累加 里插 **32 万周期**的纯 `rdcycle` 自旋，照样冻死。

**绕过手段：在 softmax / O累加 循环里按行维持 host 往返的节奏（插 htif printf）。**

## 2. 现在有两个版本，别用错

项目按精度分了目录：`fp32/`、`fp16/`、`bf16/`，各自下面再分 `C++/`（教学/参考）、`saturn/`（Saturn 裸机版）、`spike/`。

| 源文件（相对 `flash_attention/`） | 用途 | N=64 / Bc=32 | N≤63 |
|---|---|---|---|
| `fp32/saturn/saturn_flash_lean.cpp` | **无打点干净版** | ❌ 会再次停滞 | ✅ 正常 |
| `fp32/saturn/saturn_flash_lean_paced.cpp` | **带节奏打点版（v2，实测跑通）** | ✅ 已跑通 | ✅ 正常 |

对应二进制：`build/saturn_flash_lean.riscv`、`build/saturn_flash_lean_paced.riscv`

- 想拿 **N≤63** 的干净数字 → 用 `lean`。
- 想跑 **N=64** → 用 `paced`（数字仍然干净，见 §6）。
- 想在 N=64 下用**无打点**版 → 把 `BENCH_BC` 从 32 改成 **16**（列块变 partial width，绕过触发条件；代价见 §7）。

## 3. 命令（全部在 WSL，或经 Git Bash 的 `MSYS_NO_PATHCONV=1 wsl`）

同步 Windows → WSL：
```bash
cp /mnt/e/postgraduate/learning_project/flash_attention/fp32/saturn/saturn_flash_lean.cpp \
   /home/coldcane/chipyard/tests/saturn_flash_lean.cpp
# 跑 N=64 时换成 saturn_flash_lean_paced.cpp
```

编译（conda 工具链）：
```bash
cd /home/coldcane/chipyard/tests && \
/home/coldcane/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-g++ \
  -O2 -fno-common -fno-builtin-printf -march=rv64imafd_v -mabi=lp64d -mcmodel=medany \
  -specs=/home/coldcane/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/lib/htif_nano.specs \
  -static -T htif.ld saturn_flash_lean.cpp -o build/saturn_flash_lean.riscv
```
（`RWX LOAD segment` 告警是良性的。）

运行（**前台 tty 最好**；data_ok 约启动后 55 min，全程 1.5–2 h）：
```bash
cd /home/coldcane/chipyard/sims/verilator
./simulator-chipyard.harness-REFV256D128RocketConfig \
  /home/coldcane/chipyard/tests/build/saturn_flash_lean.riscv +verilator+seed+3
```

要落盘监视，**必须加 `stdbuf`**（否则 stdio 全缓冲，进程退出前日志一直是 0 字节——曾据此误判卡死 46 分钟）：
```bash
stdbuf -oL -eL ./simulator-chipyard.harness-REFV256D128RocketConfig ... > lean_run.log 2>&1
```

杀进程：
```bash
kill -9 <PID>          # 先 ps -eo pid,args | grep chipyard.harness 拿 PID
```
⚠️ **不要用 `pkill -f simulator-chipyard.harness`** —— 你的命令行本身含这个串，会把**自己的 shell** 一起杀掉。

## 4. 跑通的配置（paced / v2，三点缺一不可）

1. 在 softmax 与 O累加 循环的**每行行首**插 `wm_in()` —— **含第 0 行**：
   ```c
   for(size_t a = 0; a < br; ++a){
       if(i == 0) wm_in("sm_in", (unsigned long)i, (unsigned long)a, 1);
       ...
   }
   ```
2. `wm_in()` 的输出**逐字符复刻 `wm2()` 的格式**（`WM %-10s a=%lu b=%lu cyc=%lu dcyc=%lu`，含两次 `rdcycle`）。**格式一改短就失效** —— 见 §5 的 v1。
3. `wm_in()` 就地测出每次 printf 自耗（`c1-c0`）累计到 `g_wm_sm`/`g_wm_oa`，报告时扣回。

## 5. 五轮对照：谁行谁不行

| 版本 | 热区插了什么 | 结果 | 最后一行 |
|---|---|---|---|
| 块级打点（17 printf） | 无 | 冻结 | `jblk_mm a=32` cyc=2,106,910 |
| 子级打点（41 printf，每 4 行） | htif printf ×24 | **完整跑通** | — |
| drain（纯 `rdcycle` 自旋 20k×16） | 标量自旋 ≈32 万 cycle | 冻结 | `jblk_mm a=32` cyc=2,425,695 |
| 单 printf 最小对照 | 1 个 printf（scale 后） | 冻结 | `scale_done a=32` cyc=2,130,364 |
| **v1（每 4 行 + 短格式）** | printf 密度够、**格式变短** | 冻结 | `sc_end a=32` cyc=2,264,938 |
| **v2（每行行首 + 复刻格式）** | 密度够、格式对 | **✅ 跑通** | 正常退出 |

**读法**：
- 只有 htif printf 有效，纯 `rdcycle` 自旋无效 → 变量是 **MMIO/host 往返这个性质**，不是延时长短。
- 冻点在第 0~3 行内 → 密度必须到"每行"，且第 0 行之前就要有一次。
- printf 格式变短就失效 → 该停滞对每次 host 往返的**形状/时长**敏感。

**日志里 `a=` 的语义**（容易读错）：`iblk_*` 的 `a` 是 **i（行块）**；`jblk_*` / `sc_end` 的 `a` 是 **j（列块）**。所以 `jblk_mm a=32` = 行块 i=0 内的第二个列块，**不是 i=32**。

## 6. 跑通的数字（N=64 d=256，v2）

| 阶段 | cycle | 占比 |
|---|---:|---:|
| K 转置 | 52,384 | 2.7% |
| Kt_j 拷贝 | 22,004 | 1.1% |
| matmul (QKᵀ) | 272,513 | 14.2% |
| S *= scale | 2,080 | 0.1% |
| **softmax** | **153,124** | 8.0% |
| **O 修正 + P@V** | **861,063** | 45.0% |
| 归一化 | 19,267 | 1.0% |
| 阶段合计 | 1,382,435 | |
| **净总（减窗内打点）** | **1,915,735** | |

⚠️ 口径警告：
- 原始 `RVV cycles = 3,378,992` **不要直接用** —— 其中 1,463,257 是打点自耗（128 次窗内 printf）。softmax 原始 884,975 里 **83% 是打印开销**。
- 净总仍含 ~53 万块级 printf 开销（`c1-c0` 测不到打印后的流水线排空）。
- **不可与 N=32/48/63 的旧数字横比** —— 那几组是在完全不同的打点配置下测的。

**S 计算的口径**（对比标量版时要用）：matmul 1,048,576 次乘加 / 272,513 cycle = **3.85 MAC/cycle**，DLEN=128（4 条 fp32 通道）理论峰值 4 → **约 96% 效率**。标量版 `naive_attention` 的 `c_ns` 窗口把 **dot + scale + max** 装在一起，而 RVV 拆成了三段，同口径应为 `Kt_j 22,004 + matmul 272,513 + S*scale 2,080 = 296,597`（max 归约在 softmax 里）。K 转置（52,384）是**一次性预处理**，不属于 S 计算口径，但属于端到端口径。

## 7. 后续可选项

1. **N=64 真正干净的数字**：`BENCH_BC` 改 16，用无打点版跑。代价：块数 4→8，softmax 的按行固定开销翻倍，总时间估计 **+5%~20%**（主要涨在 softmax：行访问 128→256 次，而归约开销与元素数无关）。
2. **标量基线对比**：`fp32/saturn/saturn_flash_bench.cpp`（`BENCH_N=32`，不会停滞）会输出 `naive S 计算 cyc=` —— 拿它和 RVV 的 296,597 比，才是同口径的加速比。
3. **主线**：fp16 / bf16 量化（`fp16/saturn/saturn_flash_fp16.cpp`、`bf16/saturn/saturn_flash_bf16.cpp`），用 **N≤63** 稳跑，不受本问题影响。

## 8. 记忆索引

Claude 记忆 `n64-fullwidth-softmax-stall` 含同一结论；本手册是权威参考。
