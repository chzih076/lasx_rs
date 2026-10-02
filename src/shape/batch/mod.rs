//! 形状层的**批量样本视图**：`N` 个样本 × 固定分量数的 SOA 布局。
//!
//! 库里的批量算子（`norm3_batch`、`j2_accel_batch`、`rk4_j2_step_batch`…）都是
//! **分量拆开的数组**（x/y/z 或位置/速度各一条 `[f64]`），长度只通过一个 `n` 参数约定。
//! 这一层把"所有分量等长、且长度是 `N`"搬进类型：**构造时校验一次**，之后内核不需要再查。
//!
//! | 视图 | 分量 | 长度在哪 | 典型算子 |
//! |---|---|---|---|
//! | [`V3Ref`] / [`V3Buf`] | x、y、z（3 条） | 类型里（const `N`） | `norm3`、`unitize3`、`cross3`、`vec3_add_scaled`、`j2_accel` |
//! | [`RvMut`] | rx…vz（位置+速度，6 条） | 类型里 | `rk4_j2_step`、`ballistic_step` |
//! | [`V2Ref`] | x、y（2 条，f32） | 类型里 | `batch_distance2d` |
//! | [`V3Dyn`] / [`V3DynMut`] / [`RvDynMut`] / [`V2Dyn`] | 同上 | **运行时字段** | 同上（批量大小运行期才知道时用这一组） |
//!
//! 两组的分工同 [`crate::shape::Mat`] 与 [`crate::shape::MatDyn`]：形状编译期已知就用
//! const 泛型那组（错误能编译期报），批量大小是运行时参数就用 `*Dyn` 那组（构造时校验一次）。
//!
//! # 为什么要有这一层
//!
//! **发布构建里长度不一致原本是 UB**：内核用第一条数组的长度当 `n`
//! （`let n = xs.len();` 之后按 `n` 读 `ys`/`zs`/`out`），而 FFI 转发层只有
//! `debug_assert_eq!`（发布构建下编译掉）。`V3Ref::new` 把这件事变成**一次显式校验**：
//! 三条不等长就返回 `Err`，拿不到视图，也就调不出内核。
//!
//! # 谁守什么
//!
//! | 契约 | 由什么保证 |
//! |---|---|
//! | 每条分量长度都等于 `N` | **类型系统**：`*::new` 校验一次，之后 `N` 是类型参数 |
//! | 输入与输出不重叠 | **借用检查器**：输入是 `&V3Ref`（共享借用），输出是 `&mut V3Buf`（独占） |
//! | 数值与直接调内核一致 | **同一个内核**（这一层只做视图与转发），测试用 `to_bits()` 锁定 |
//! | 对齐 | [`AlignedVec`](crate::aligned::AlignedVec)（与 `api::*` 输出同一条约定） |
//!
//! # 例
//!
//! ```
//! use lasx_rs::shape::batch::{V3Buf, V3Ref};
//!
//! const N: usize = 2;
//! let x = [3.0f64, 0.0];
//! let y = [4.0f64, 1.0];
//! let z = [0.0f64, 0.0];
//! let v = V3Ref::<f64, N>::new(&x, &y, &z)?;
//! let mut mag = lasx_rs::shape::VecBuf::<f64, N>::new();
//! v.norm3_into(&mut mag);
//! assert_eq!(mag.as_slice(), &[5.0, 1.0]);
//! # Ok::<(), lasx_rs::api::Error>(())
//! ```

pub mod dist2d;
pub mod dynamic;
pub mod rv;
pub mod v3;

pub use dist2d::V2Ref;
pub use dynamic::{RvDynMut, V2Dyn, V3Dyn, V3DynMut};
pub use rv::RvMut;
pub use v3::{V3Buf, V3Ref};
