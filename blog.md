# 把 Silero VAD 从 70 us 优化到 19.6 us：一次 TileLang 端到端优化实践

这篇文章记录一次真实的小模型推理优化：在 Jetson AGX Orin 上，把 Silero
VAD 的单窗口延迟从约 70 us 降到 19.6 us。最终实现保持全程 FP32，和 ONNX
Runtime FP32 的概率最大绝对误差为 2.801e-6，低于 1e-5 的目标。

文章面向已经了解 CUDA block、warp、shared memory 和算子融合，但缺少系统性能
优化经验，或者刚接触 TileLang 的读者。重点不是复述最终代码，而是解释每一步为什么
有效、哪些尝试失败了，以及怎样避免测出一个很快但错误的结果。

## 1. 问题长什么样

固定 16 kHz 的 Silero VAD 每次处理 512 个新采样，也就是 32 ms 音频。模型前向为：

```text
context[64] + window[512]
    -> reflect pad 到 640
    -> STFT: 258 x 256 basis
    -> Conv1d x 4
    -> LSTM cell: hidden size 128, 4 x 128 gates
    -> FC + sigmoid
    -> speech probability
```

这个模型的计算量不大，却是一个典型的低延迟难题：工作负载小到 GPU 很快就能算完，
kernel launch、同步、几百字节的数据传输和 CPU 侧准备工作反而会成为主角。同时 LSTM
是有状态的，第 `n+1` 个窗口依赖第 `n` 个窗口的 `h/c`，不能简单把很多窗口组成 batch。

最终结果如下。表里的延迟都是 Rust 实际调用路径上的每窗口 p50，而不是只测 kernel
内部的 CUDA event 时间。

| 阶段 | 端到端 p50 | 关键变化 |
|---|---:|---|
| 7-kernel FP32 链 | 约 70 us | 每层独立 launch |
| 初版 fused cooperative kernel | 约 47.5 us | 7 次 launch 合为 1 次 |
| warp FP32 归约和 mapped output | 约 29.6 us | 消除串行点积和 D2H |
| persistent，直接读取 mapped input | 约 21.9 us | 每个文件只 launch 一次 |
| persistent + 正确的 input cache | 约 21.0 us | 避免 32 blocks 重复读取 host mapping |
| 256 threads + CPU 热路径精简 | **19.5-19.7 us** | 重阶段增加 warp，直接构造 mapped input |

## 2. 先统一测量口径

优化前最重要的工作不是改 kernel，而是决定“延迟”到底指什么。只测 GPU kernel duration
很容易得到一个漂亮但对业务无用的数字。本项目采用的重复路径包括：

1. 构造 `context + window + reflect padding`。
2. CPU 发布新输入。
3. GPU 完整执行 STFT、4 层卷积、LSTM 和 FC。
4. CPU 等到概率输出可见。
5. 更新下一窗需要的 64-sample context。

persistent kernel 的一次性启动发生在文件开始处，不计入每窗口 p50，但计入完整文件的
实际运行时间。ffmpeg 解码和 VAD 后的分段状态机不是单次模型 forward 的组成部分。

Rust 在调用 `forward` 前后直接计时：

```rust
let t0 = std::time::Instant::now();
let pr = self.forward(&buf, (probs.len() + 1) as i32)?;
times_ns.push(t0.elapsed().as_nanos() as u64);
```

建议同时保留分阶段计时，但不要把阶段计时之和误当成最终指标。计时本身、CPU 调度和
cache 状态都会产生扰动。最终版本连续 5 次的 p50 为 19.5-19.7 us，p99 约为
19.8-20.3 us。

## 3. TileLang 最小语法地图

先用最终 kernel 的骨架认识本文会用到的 TileLang 原语：

```python
@T.prim_func
def vad_staged(...):
    with T.Kernel(32, 1, threads=256) as (bx, by):
        weight_sh = T.alloc_shared((ROWS_PER_BLOCK, 128), "float32")
        acc = T.alloc_local((1,), "float32")

        for row, k in T.Parallel(ROWS_PER_BLOCK, 128):
            weight_sh[row, k] = weight_global[bx * ROWS_PER_BLOCK + row, k]

        T.sync_threads()

        for it in T.serial(n_windows[0]):
            # 各 stage 的计算
            T.sync_grid()
```

需要记住五点：

- `T.Kernel(grid_x, grid_y, threads=...)` 定义 CUDA grid 和 block。
- `T.Parallel` 把逻辑迭代映射到线程；迭代空间比线程多时，每个线程会处理多个元素。
- `T.serial` 明确保留串行循环，适合小循环、固定顺序归约或 persistent 时间循环。
- `T.alloc_local` 通常落到寄存器，`T.alloc_shared` 对应 block shared memory。
- `T.sync_threads()` 只同步一个 block；`T.sync_grid()` 需要 cooperative launch，并要求
  整个 grid 能同时驻留，否则可能无法启动或死锁。

TileLang 降低了表达 CUDA kernel 的成本，但不会替你决定并行分解、内存层级和同步位置。
性能仍然来自这些选择。

## 4. 从 70 us 到 47 us：基础优化比花哨技巧更重要

### 4.1 先确认是不是 launch-bound

早期实现把前向拆成 7 个 kernel：STFT、4 个 Conv1d、LSTM 矩阵部分和 gate/FC。
单个 kernel 的计算不重，但这台 Jetson 上一次 dispatch 的下限约为 5 us，因此 7 次
launch 仅 dispatch 就接近 35 us。

这里做过一个很重要的反例验证：CUDA Graph replay 没有降低这台驱动上的 7 次 dispatch
floor。Graph 能减少 CPU 提交工作，不保证底层设备调度开销也被压成一次。优化不能依赖
“Graph 理论上应该更快”，必须用近空 kernel 和完整链分别测量。

### 4.2 融合不等于塞进一个 block

最直接的融合版本把所有工作放进一个 block，结果约 262 us，比 7-kernel 链更慢。原因
不是融合本身，而是它只使用一个 SM，却需要反复读取约 1.2 MB 权重。并行度被人为压扁，
权重带宽和长串行归约占满了单个 SM。

真正有效的方案是 multi-block cooperative fusion：

```python
G = 32

with T.Kernel(G, 1, threads=128) as (bx, by):
    # 每个 block 负责连续的一段输出
    output = bx * ROWS_PER_BLOCK + local_row

    # STFT
    ...
    T.sync_grid()

    # Conv0
    ...
    T.sync_grid()

    # Conv1 -> Conv2 -> Conv3 -> LSTM -> FC
```

各 stage 的中间结果放在 global buffer 中，例如 `spec_g`、`e0_g`、`feat_g` 和 `z_g`。
block 按输出行做连续分区，避免写冲突，也使权重访问更容易合并。stage 之间使用
`T.sync_grid()`，保证后一层看到前一层所有 block 的结果。

这个设计有三个基础收益：

1. 7 次设备 dispatch 变成 1 次 cooperative launch。
2. 所有 SM 都能参与一个小模型的各个 stage。
3. 中间结果留在 GPU，不再经过 host，也不需要 7 个独立执行边界。

### 4.3 显式 padding，换掉热循环里的边界逻辑

卷积输入使用带零边界的固定形状 global buffer：

```python
spec_g: [129, 6]  # 有效 4 帧，两侧各一个 0
e0_g:   [128, 6]
e1_g:   [64, 4]
e2_g:   [64, 3]
```

每层在写有效输出的同时补好两端的 0。下一层的内循环就可以直接读取 `t + k`，不需要
每次乘加都判断 padding。小模型中，减少热循环里的分支和地址逻辑往往比省几个全局
元素更值钱。

### 4.4 权重常驻、状态原地更新、参数数组复用

模型权重只在 `VadModel` 初始化时上传一次，运行时只传 device pointer。LSTM 状态也不再
在两个 buffer 之间 ping-pong：Kz 在 gate stage 写入前读完旧 `h`，gate 对每个索引独占
更新 `h/c`，因此可以安全原地更新。

CUDA driver API 的 `kernelParams` 也在加载时构造一次：

```rust
let slot_bytes: Vec<[u8; 8]> = spec.slots.iter()
    .map(|&s| ptr[slot_index(s)].to_le_bytes())
    .collect();

self.kparams = self.slot_bytes.iter_mut()
    .map(|b| b.as_mut_ptr() as *mut c_void)
    .collect();
```

这些工作单独看都不大，但对于几十微秒的目标，任何每窗重复的分配、参数打包和状态复制
都值得消除。

### 4.5 mapped input 消除小 H2D

输入只有 640 个 FP32，显式 `cuMemcpyHtoD` 的固定成本比数据本身更显眼。Jetson 是共享
物理内存架构，因此使用 `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED`
分配 CPU 写、GPU 读的 mapped buffer：

```rust
// DEVICEMAP(2) | WRITECOMBINED(4)
cuMemHostAlloc(&mut raw, 640 * 4, 6);
cuMemHostGetDevicePointer(&mut xpad_dev, raw, 0);
```

write-combined 页适合 CPU 顺序写、GPU 读取，CPU 不应频繁回读。这个技巧高度依赖平台；
在独立显卡和 PCIe 系统上，mapped host read 可能远慢于正常 device memory，不能照搬。

完成这些基础优化后，端到端 p50 从约 70 us 降到 47.5 us。这个阶段最大的经验是：先
消灭固定开销和明显的串行边界，再谈指令级优化。

## 5. 从 47 us 到约 30 us：warp-per-output FP32 归约

初版 fused kernel 虽然解决了 launch 问题，但一个线程仍会串行完成一整行点积。例如
STFT 的每个输出需要 256 项，LSTM Kz 的每行需要两个 128 项点积。线程级并行很多，
单个输出内部却很串行。

更合适的分解是一个 warp 负责一个输出。以 Kz 为例：

```python
for warp, lane in T.Parallel(8, 32):
    for q in T.serial(2):
        row = bx * 16 + warp * 2 + q
        a1 = T.alloc_local((1,), "float32")
        a2 = T.alloc_local((1,), "float32")
        a1[0] = T.float32(0)
        a2[0] = T.float32(0)

        for kk in T.serial(4):
            k = lane + kk * 32
            a1[0] += Wsh[warp * 2 + q, k] * fsh[k]
            a2[0] += Rl[row, k] * hsh[k]

        for s in T.serial(5):
            delta = 16 >> s
            a1[0] += T.shfl_down(a1[0], delta)
            a2[0] += T.shfl_down(a2[0], delta)

        if lane == 0:
            z_g[row] = a1[0] + a2[0] + Bl[row]
```

每个 lane 只处理 4 项，再通过 `T.shfl_down` 做 warp tree reduction。相比 shared-memory
归约，它不需要反复写 shared 和 `sync_threads`，并且 warp shuffle 的通信路径固定。

STFT 同理：256 项被 32 个 lane 分成每 lane 8 项。Conv0 的输入通道是 129，每 lane
处理最多 5 项。

### 5.1 卷积的三累加器 ILP

卷积核宽度是 3。不要把三个 tap 全塞进同一个长依赖链，可以使用三个独立累加器：

```python
a0[0] += wsh[oi, ic, 0] * xsh[ic, t]
a1[0] += wsh[oi, ic, 1] * xsh[ic, t + 1]
a2[0] += wsh[oi, ic, 2] * xsh[ic, t + 2]
```

GPU 可以交错调度三条 FMA 依赖链，暴露更多指令级并行。warp reduction 后再按固定顺序
计算 `a0 + a1 + a2 + bias`。

### 5.2 为什么没有直接用 GEMM/TF32

目标要求概率误差小于 1e-5。sm_87 上把这些点积交给默认 GEMM 路径，可能使用 TF32
tensor core，误差预算不够稳定。因此最终版本保留显式 FP32 乘加和 FP32 shuffle 归约。

归约树会改变浮点加法顺序，所以“使用 FP32”不代表逐 bit 相同。策略是：

- 每次改变归约方式后都跑完整有状态序列，而不是只测单层随机输入。
- FC 最后的 128 项仍保留原始顺序串行求和，避免在概率输出前再放大一次顺序误差。
- 把 1e-5 当硬门槛，不用“分段看起来一样”代替数值验证。

这组 warp 归约和 shared staging 把端到端延迟进一步压到约 30 us，同时概率误差仍在目标内。

## 6. 从每窗 launch 到 persistent kernel

即使已经融合为一个 kernel，每个 32 ms 窗口仍要 launch 一次。计算只有几十微秒时，
这次 launch 依然不可忽略。VAD 又天然是串行状态机，因此适合 persistent kernel：每个
文件只 launch 一次，kernel 内部循环等待 CPU 发布下一窗。

```python
with T.Kernel(32, 1, threads=256) as (bx, by):
    # 只加载一次，跨所有窗口复用
    for row, k in T.Parallel(ROWS_PER_BLOCK, 128):
        Wsh[row, k] = Wl[bx * ROWS_PER_BLOCK + row, k]
    T.sync_threads()

    for it in T.serial(n_windows[0]):
        wait_for_ready(it)
        run_stft_conv_lstm_fc()
        publish_done(it + 1)
```

把 basis、卷积权重和一部分 LSTM W 的 shared staging 放到窗口循环外非常关键。否则只是
省掉 launch，却仍在每窗重复装载所有权重，persistent 的收益会被吃掉。

### 6.1 cooperative persistent kernel 的资源约束

普通 kernel 可以让未调度 block 等前面的 block 退出。persistent cooperative kernel
中，所有 block 会反复执行 `sync_grid`，因此整个 grid 必须同时驻留。

最终 cubin 的资源为：

```text
grid              32 blocks
threads/block     256
registers/thread  82
dynamic shared    45,904 bytes/block
```

在目标 Orin 上该配置可以 cooperative launch，并让 32 个 block 覆盖所有 SM。调大线程数
或 shared memory 前必须重新检查资源，否则“理论并行度更高”可能直接使 cooperative
launch 失败。

## 7. 最危险的 bug：快了 2 us，却滞后一窗

persistent kernel 需要 CPU 和 GPU 跨设备握手。最初的 cache 优化让 block 0 把 mapped
`xpad` 复制到 device `xpad_cache`，再让所有 block 从 device cache 读取。延迟一度进入
18-19 us，但输出序列是：

```text
ONNX:    w0, w1, w2, ...
错误版:  w0, w0, w1, ...
```

JFK 样例的首段也从正确的 `0.32 -> 2.27` 整体后移为 `0.35 -> 2.30`。这是非常典型的
“benchmark 通过、模型语义失败”：每次 forward 都返回一个合法概率，甚至序列形状和
数值范围都正常，只是读到了上一窗。

### 7.1 Release/Acquire 不只是一对关键字

CPU 侧先写输入，再 release 发布序号：

```rust
write_xpad();
ready.store(seq, Ordering::Release);
```

GPU 侧 acquire 等待：

```python
seen = T.atomic_load(ready, memory_order="acquire")
while seen <= it:
    seen = T.atomic_load(ready, memory_order="acquire")
```

但 TileLang 默认 atomic 使用 `cuda::thread_scope_device`，它不能自动成为可靠的 CPU/GPU
system-scope 握手。因此编译阶段为控制变量注入了 system-scope atomic：

```cpp
cuda::atomic_ref<int, cuda::thread_scope_system> ref(*ptr);
int seen = ref.load(cuda::memory_order_acquire);
```

即使控制变量使用 system scope，也不能想当然地认为随后由其它线程执行的普通 mapped
load 已经获得正确数据。目标平台上，block 0 在观察 `ready` 后立刻复制输入，仍可能读到
旧 cache line。

### 7.2 最终采用的发布顺序

最终 kernel 使用两次 grid barrier：

```python
if bx == 0:
    # 一个线程 system-acquire ready
    while T.atomic_load(ready, memory_order="acquire") <= it:
        pass

T.sync_grid()  # 把本轮发布状态带到整个 cooperative grid

if bx == 0:
    for i in T.Parallel(640):
        xpad_cache[i] = xpad[i]

T.sync_grid()  # 所有 block 等待 device cache 填充完成
```

第一道 barrier 之后才读取 mapped input，第二道 barrier 广播 cache 完成。这个顺序在目标
Jetson 上经过逐窗概率和分段边界验证。若要移植到独立显卡或不同一致性模型，优先考虑
显式异步 copy + event/stream 依赖，不要把这里的 unified-memory 行为当成跨平台保证。

这次失败带来的通用经验是：跨 CPU/GPU 发布协议必须同时验证控制变量和普通数据的
可见性；一个正确变化的 sequence number 不代表旁边的数据一定是同一代。

## 8. 128 threads 到 256 threads：减少 warp 的串行批次

在约 21 us 时，输入和 host 开销已经很小，主要时间都在 GPU 完成等待。资源检查显示
128-thread block 仍有增加 warp 的空间，于是把 block 提到 256 threads，也就是从 4 个
warp 增加到 8 个 warp。

重点不是简单修改 `threads=256`，还要重写逻辑任务到 warp 的映射。以 Kz 的 16 行/block
为例：

```python
# 128 threads: 4 warps，每个 warp 串行 4 行
for warp, lane in T.Parallel(4, 32):
    for q in T.serial(4):
        row = warp * 4 + q

# 256 threads: 8 warps，每个 warp 串行 2 行
for warp, lane in T.Parallel(8, 32):
    for q in T.serial(2):
        row = warp * 2 + q
```

STFT 和 Conv0 做了相同调整。每行内部仍是相同的 lane 分工和 shuffle reduction，因此没有
改变单行的 FP32 归约方式。较小的 Conv1/2/3 没有足够独立输出喂满 8 个 warp，保持原来
映射反而更简单。

编译后寄存器只从 80 增加到 82，shared memory 不变，32-block cooperative launch 仍然
成功。这个改动把 p50 从约 21.0 us 降到 19.9 us 左右。

这里的经验是：block size 不是一个孤立的调参旋钮。增加线程只有在你同步减少了关键路径
上的 `T.serial` 批次时才有意义，否则只是增加空闲线程和资源压力。

## 9. 最后 0.3 us 来自 CPU 热路径

原 Rust 路径先在栈上构造 640-float `xpad`，再把它完整复制到 mapped buffer：

```rust
let mut xpad = [0f32; 640];
build_context_window_and_reflect_pad(&mut xpad);
copy_nonoverlapping(xpad.as_ptr(), self.xpad_host, 640);
```

最终版本直接写 mapped write-combined 页：

```rust
std::ptr::copy_nonoverlapping(
    self.context.as_ptr(), self.xpad_host, CONTEXT,
);
std::ptr::copy_nonoverlapping(
    window.as_ptr(), self.xpad_host.add(CONTEXT), WINDOW,
);

for i in 0..CONTEXT {
    *self.xpad_host.add(INPUT_LEN + i) = window[WINDOW - 2 - i];
}

ready.store(seq, Ordering::Release);
```

反射区的源索引可以直接落到 `window[510..447]`，不需要回读 write-combined 页。这样省掉
一次 2.5 KB 中间复制，输入准备约 0.2 us，最终 p50 稳定到 19.5-19.7 us。

输出也使用 mapped memory。GPU 先写 `prob`，再 system-release `done`；CPU acquire 观察
`done >= seq` 后读取概率：

```rust
let prob = loop {
    if done.load(Ordering::Acquire) >= seq {
        break std::ptr::read_volatile(self.prob_host);
    }
    std::hint::spin_loop();
};
```

对于 20 us 级任务，spin wait 比线程休眠或复杂通知机制更合适，因为等待极短且调用本身
就是同步、有状态的。但这会占用一个 CPU core，若服务需要大量并发模型实例，应重新评估。

## 10. 精度验证必须和性能验证并行进行

最终生产路径使用与 Rust 完全相同的 PCM16 输入，逐窗运行 ONNX Runtime FP32 和 TileLang
kernel，共比较 JFK 样例的 344 个有状态窗口：

```text
max_abs_error  = 2.80100882e-6
p99_abs_error  = 2.41816992e-6
mean_abs_error = 1.70915890e-7
```

分段边界也必须作为第二层验证：

```text
#1  0.32 -> 2.27
#2  3.27 -> 4.45
#3  5.38 -> 7.68
#4  8.16 -> 11.00
```

为什么既比较概率又比较边界？概率误差能发现数值退化，边界能发现状态错位、context 更新
错误和最后一窗 padding 错误。前面那次滞后一窗的 cache bug，单看最大概率范围无法发现，
但逐窗对齐和边界立刻会失败。

## 11. 如何复现实验

核心 TileLang 实现在 `../tilelang-poc/vad/kernels6.py` 的 `vad_staged`，Rust 端到端
调用在 `src/vad.rs`。修改 kernel 后重新生成嵌入产物并构建：

```bash
python3 scripts/build_vad_kernels.py
cargo build --release
```

检查 cubin 的寄存器使用，并确认 metadata 中的 grid、block 和 shared memory 与预期一致：

```bash
cuobjdump --dump-resource-usage assets/k_vad_staged.cubin
sed -n '1,30p' assets/k_vad_staged.json
```

运行 Rust 实际端到端 benchmark：

```bash
OTOGRAPH_VAD_BENCH=1 \
  target/release/otograph --dry-run /path/to/audio.flac
```

最终版本在目标 Orin 上的典型输出为：

```text
[vad-bench] 344 windows mean=19636ns (19.6us) p50=19.6us p99=20.0us min=19.1us
[vad-bench] phase us: mapped-input=0.2 launch=0.0 mapped-output-wait=19.2
```

精度验证应使用同一份 PCM 输入，把 ONNX 的 `state_out` 逐窗传给下一窗，同时让 TileLang
kernel 保持自己的 `h/c`，然后按窗口索引比较概率。不要为方便而把每窗 state 清零，也
不要分别走两套音频解码流程，否则测到的是不同输入或无状态模型。

## 12. 一套可复用的优化顺序

面对类似的小模型、低延迟 GPU workload，可以按下面顺序推进：

1. **固定验收口径。** 明确是否包含输入准备、launch、同步、输出和状态更新。
2. **建立逐窗 golden。** 有状态模型必须比较完整序列，不能只测单个随机输入。
3. **先画时间预算。** 区分 dispatch-bound、memory-bound 和 arithmetic-bound。
4. **消除固定开销。** 权重常驻、参数复用、状态原地更新、小 copy 合并。
5. **再做融合。** 先验证单 block 是否丢失并行度，再决定 multi-block cooperative 设计。
6. **为归约选择合适粒度。** 小矩阵常常更适合 warp-per-output，而不是通用 GEMM。
7. **shared memory 只缓存真正复用的数据。** 同时检查 shared 和寄存器对 occupancy 的影响。
8. **减少关键路径的串行批次。** 增加 block threads 时同步调整逻辑 warp 映射。
9. **把同步当作算法的一部分。** 删除 barrier 前先写清生产者、消费者和可见性关系。
10. **每个性能改动后先验精度，再记性能。** 错误结果的 18 us 没有任何价值。

## 13. TileLang 初学者最容易踩的坑

### `T.Parallel` 不代表自动得到好访存

循环轴的顺序会决定线程访问哪个维度。让相邻线程访问连续权重行或连续 `k`，通常比单纯
扩大并行迭代空间更重要。编译后仍应查看生成 CUDA 或 SASS，而不是只读 DSL。

### `T.sync_grid` 不是更大的 `sync_threads`

它要求 cooperative launch 和全 grid 同驻留。每次改变 block size、寄存器或 shared
memory，都要重新检查资源并实际启动。

### shared memory 不是越多越快

本例把高复用的 basis、卷积权重、输入 tile 和部分 LSTM W 放到 shared。若把全部权重都
硬塞进去，会降低 occupancy 或超过容量。缓存选择应由“跨多少次使用”决定。

### 浮点优化不能只看 dtype

FP32 乘加、TF32 tensor core、串行求和和 tree reduction 都可能给出不同结果。精度约束
最终作用在模型输出，不是某个 kernel 的声明 dtype。

### mapped memory 的正确性和性能都依赖平台

Jetson unified memory 上有效的 zero-copy，不一定适合 PCIe GPU。跨 CPU/GPU 的普通数据
发布也不能只靠一个 device-scope atomic。先理解 memory scope，再做性能判断。

## 结语

这次优化没有依赖一个神奇的 schedule。70 us 到 47 us 主要来自系统层面的基础工作：
减少 launch、保持多 SM 并行、让权重和状态常驻、消除小 copy。47 us 到 30 us 来自
warp 级 FP32 归约和 shared staging。30 us 到 20 us 则来自 persistent execution、严格
的 CPU/GPU 发布协议、更高的 block 并行度，以及最后几百纳秒的 host 热路径整理。

最值得保留的经验是：低延迟优化是一项端到端工程。kernel 算得快只是必要条件；launch、
内存一致性、状态顺序、数值误差和调用端代码共同决定最终结果。只有在同一套真实口径下，
同时守住正确性和性能，19.6 us 才是一个可以交付的数字。
