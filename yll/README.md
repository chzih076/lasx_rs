# lasx — YouLiLong 原生扩展

把 [`lasx_rs`](../) 的龙芯 LASX(256 位)/LSX(128 位) 向量内核暴露给 YouLiLong 脚本。
Rust 实现，产出 `liblasx.so`，按
[`C-C++原生扩展开发指南`](../../docs/高级特性/C-C++原生扩展开发指南.md) 的约定接入。

## 构建与运行

```bash
# 在 lasx_rs 仓库根
cargo build --release -p lasx_yll
cp target/release/liblasx.so yll/

# 用 YouLiLong 解释器跑端到端测试（成功路径 + 错误上抛路径）
<YouLiLong>/target/release/youli_long yll/test_lasx.yli
```

## 接口

| 函数 | 说明 |
|---|---|
| `dot(a, b)` | f32 点积 → Float |
| `sum(x)` | f32 归约 → Float |
| `dot_i8(a, b)` | int8 量化点积 → Int（元素需在 −128..=127） |
| `matmul(a, b, m, k, n)` | `C[m×n] = A[m×k]·B[k×n]`，A/B 为行主序扁平数组 → Array |
| `norm3(xs, ys, zs)` | `out[i] = √(x²+y²+z²)`（f64）→ Array |
| `distance2d(px, py, xs, ys)` | 各点到 `(px, py)` 的距离（f32）→ Array |
| `vec3_add_scaled(ax, ay, az, bx, by, bz, s)` | `o = a + s·b` → `[ox, oy, oz]` |
| `j2_accel(rx, ry, rz, mu, j2, re)` | 中心引力 + J2 摄动加速度 → `[ax, ay, az]` |
| `rk4_step(rx, ry, rz, vx, vy, vz, mu, j2, re, dt)` | 批量 RK4 J2 单步（**单线程**）→ `[rx, ry, rz, vx, vy, vz]` |
| `propagate(rx, ry, rz, vx, vy, vz, mu, j2, re, dt, steps)` | 批量 RK4 J2 多步传播（**多核常驻池**，数组只转换一次）→ 终态 `[rx, ry, rz, vx, vy, vz]` |

常量：`VERSION`、`ALIGN`（LASX 建议的缓冲对齐字节数）。

```youlilong
use "./lasx"

a = [1.0, 2.0, 3.0, 4.0]
print(lasx.dot(a, [1.0, 1.0, 1.0, 1.0]))   // 10
print(lasx.norm3([3.0], [4.0], [0.0]))     // [5]
```

## 多步传播：用 `propagate`，别在脚本里循环

两者数值**逐位一致**（同样的内核、同样的每步顺序，池只改下标切分），差别只在调用策略：

```youlilong
// A) 脚本循环：单线程，而且**每一步**都要把 6 个数组在脚本值与 Rust 缓冲之间搬一遍
cur = lasx.rk4_step(rx, ry, rz, vx, vy, vz, mu, j2, re, dt)
for s in 1..200 {
    cur = lasx.rk4_step(cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], mu, j2, re, dt)
}

// B) 一次到位：常驻线程池多核，数组只转换一次
final = lasx.propagate(rx, ry, rz, vx, vy, vz, mu, j2, re, dt, 200)
```

`yll/bench_propagate.yli` 的实测（n = 20000 星 × 200 步）：

| 写法 | 总时间 | 每步 | 相对 |
|---|---|---|---|
| A 脚本循环 `rk4_step` | 5608.8 ms | 28.04 ms | 1× |
| B `propagate` | 50.9 ms | 0.254 ms | **110×** |

（重复测量在 **99–112×** 之间：A 列对机器负载很敏感，B 列基本不动。）

这个 110× 远大于内核本身的多核加速（6.9×），因为 A 的每步里**内核只占约 0.6 ms**，
其余 27 ms 全是"把 12 万个数值在脚本值与 Rust 缓冲之间来回搬"。结论：
**脚本侧循环调用批量内核时，真正的代价不是内核，而是每次调用的数据搬运**——
把循环放进库里（`propagate` 就是这么做的）比在脚本里优化循环有效得多——量级差别，不是调优。

> 这个对照脚本要跑约 6 秒，**不进 CI**，手动执行：
> `youli_long yll/bench_propagate.yli`

## 错误处理：一律上抛

**所有校验失败都返回 `yll_error(...)`**，解释器会抛成可 `try/catch` 的运行时错误——
不静默返回 0、不吞掉错误、不产生半个结果：

```youlilong
try {
    lasx.dot([1.0, 2.0], [1.0])            // 长度不一致
} catch (e) {
    print(e)   // 无效操作: a 与 b 长度不一致：2 vs 1
}

try {
    lasx.matmul([1.0,0.0,0.0,1.0], [1.0,2.0,3.0,4.0], 3, 3, 3)
} catch (e) {
    print(e)   // 无效操作: a 的形状不对：声明 3×3=9 个元素，实际 4
}

try {
    lasx.j2_accel([7.0e6], [0.0], [0.0], 0.0, 1.08262668e-3, 6.378137e6)
} catch (e) {
    print(e)   // 无效操作: j2_accel: 常数必须为正
}
```

会报错的输入：

- 参数不是数组 / 元素不是数值；
- 参与同一运算的数组长度不一致；
- `matmul` 的数组长度与声明的 `m/k/n` 不自洽；
- `int8` 元素越界（不在 −128..=127）；
- 物理常数非法（`mu <= 0`、`re <= 0`、`j2`/`dt` 为非有限值）；
- `propagate` 的 `steps < 0`，或位置与速度数组长度不一致；
- 数组元素数超过 `1 << 28`。

错误消息**具体到参数名与下标**（如 "参数 `x` 的第 1 个元素不是数值"），
便于脚本侧直接定位。

## 实现要点

- **分层校验**：扩展层判"是不是数组、元素是不是数值、数组长度与形状是否自洽"，
  库的 `lasx_*_checked` 再兜住结构性错误（空指针、负长度、相乘溢出）。两层错误
  统一成同一种 `yll_error`。
- **对齐**：脚本数组进来后先拷贝进 32 字节以上对齐的缓冲（`AlignedVec`）再调内核。
  LASX 是 32 字节访存，未对齐会跨缓存行——实测 glibc `malloc`/Dart FFI 只有约一半
  落在 32 字节边界，所以这一步不是多余的。
- **几乎没有共享状态**：除 `propagate` 外都是纯函数（入参拷贝进 `AlignedVec`、
  无共享可变状态），解释器不给 GIL 也不影响它们。`propagate` 持有一个进程内常驻
  线程池（跨调用复用才有多核收益），池的派活是独占的，故用 `Mutex` 串行化并发调用；
  上一次调用若 panic，池会先让 worker 全部收工再续抛，状态仍干净，`Mutex` 中毒也不
  影响继续使用。
- **不泄漏**：字符串按"调用期间存活"借出（参考实现用 `CString::into_raw()` 每次调用
  泄漏一次），本扩展不回退到那种写法。

## 目录

| 文件 | 说明 |
|---|---|
| `src/lib.rs` | 入口 `yll_init_lasx` 与函数/常量注册 |
| `src/yll.rs` | `yll.h` 的 extern 声明与参数读取辅助 |
| `src/convert.rs` | 脚本数组 ↔ `AlignedVec` 转换 |
| `src/funcs.rs` | 各内核的对外函数（校验 + 调库 + 造返回值） |
| `lib.ylh` | 接口声明（LSP/IDE 用） |
| `youli.yaml` | 包配置（含 `native:` 段） |
| `test_lasx.yli` | 端到端测试：成功路径 + 10 条错误上抛路径 |
| `bench_propagate.yli` | 调用策略对照（脚本循环 vs `propagate`，手动跑，不进 CI） |
