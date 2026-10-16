//! **表达式级代数重写：模式识别 + 安全区分析（原型，本轮不生成代码）**。
//!
//! # 本轮边界（**如实**）
//! 本模块只做**识别与判定**（"这段公式能不能重写成递推、安不安全"），**不生成改写后的代码**、
//! **不接入宏层** ⇒ **默认路径行为完全不变**（现有 `matmul!`/拉普拉斯入口一字未改）。
//! 宏层集成（把判定结果变成递推代码 + `#[fast_trig]` 属性）留独立一轮。
//!
//! # 三条规约（识别条件 / 改写规则 / 不认识就不改写）
//! 1. **识别条件**（全部满足才算匹配）：
//!    - 索引必须**等距**且**步长是编译期字面量**：`t_i = base + i·step`（`step` 为字面量）；
//!    - 被调函数必须是 `exp`/`cos`/`sin`，且其**参数是索引的线性函数**（`a + i·b`）；
//!    - **行可分**：每行一个 `(σ_j, ω_j)`，行内 `t_i` 等距（`e^{−σ_j t_i}`、`cos/sin(ω_j t_i)`）；
//!    - 参数是**运行期量**（非字面量）时：允许，但**只能生成运行期检查 + 逐行回退**，不能编译期定。
//! 2. **改写规则**：等距 ⇒ `e^{−σΔt}` 逐次乘、`cos/sin(ωΔt)` 旋转递推（每元素 ~2 FMA、**零超越调用**）。
//! 3. **不认识就完全不改写**：`Varies`（步长非常量/非等距索引/非线性参数）⇒ **`Rewrite::None`**，
//!    **不静默改变语义**（负例见 `tests`）。
//!
//! # 安全区（**与库内同源**）
//! 判据直接调用 [`crate::laplace::recurrence_safe`]，阈值用
//! [`crate::laplace::RECURRENCE_MAX_PHASE`] / [`crate::laplace::RECURRENCE_MAX_PHASE_STEP`] ——
//! **同一处定义、不各写一份**（证据：测试 `safe_zone_uses_library_constants`）。
//!
//! # 契约
//! 重写路径**与 libm 逐元素路径不逐位一致** ⇒ **必须显式 opt-in**（宏级属性草案 `#[fast_trig]`）；
//! **默认仍 libm/位精确**。误差界与安全区见 `docs/ops.md` §9.10.1。

use crate::laplace::{RECURRENCE_MAX_PHASE, RECURRENCE_MAX_PHASE_STEP, recurrence_safe};

/// 被调函数（目前只认这三个）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrigOp {
    /// `exp(·)`
    Exp,
    /// `cos(·)`
    Cos,
    /// `sin(·)`
    Sin,
}

/// 索引步长的来源（决定**编译期**还是**运行期**判定安全区）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepKind {
    /// **字面量**（编译期可知）⇒ 编译期判安全区。
    Literal(f64),
    /// 运行期常量（同一值、但编译期未知）⇒ 生成运行期检查 + 逐行回退。
    RuntimeConst,
    /// **非常量/非等距**（每步都可能不同）⇒ **不匹配、不改写**。
    Varies,
}

/// 一元超越调用在等距索引上的模式描述（宏层会把 AST 降成这个描述）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelPattern {
    /// `t_i = base + i·step` 的 `base`。
    pub base: f64,
    /// 步长来源。
    pub step: StepKind,
    /// 元素个数 `N`。
    pub n: usize,
    /// 被调函数。
    pub op: TrigOp,
    /// 参数是否**只由索引线性构成**（`arg = a + i·b`，`b` 与索引步长同源）。
    pub lin_in_index: bool,
    /// **显式 opt-in**（宏级属性 `#[fast_trig]` 的等价物）：**未挂属性 = `false` ⇒ 完全不重写**。
    pub fast_trig_opt_in: bool,
}

/// 判定结果。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rewrite {
    /// 可改写为递推（每元素 ~2 FMA、零超越调用）。`compile_time_safe` = 编译期即可判定安全区。
    Recurrence { compile_time_safe: bool },
    /// **不匹配 / 不安全 ⇒ 完全不改写**（走 libm，语义不变）。
    None,
}

/// **识别 + 判定**（纯函数；不生成代码）。
///
/// - `Varies` ⇒ [`Rewrite::None`]；
/// - 非 `lin_in_index`（参数非线性）⇒ [`Rewrite::None`]；
/// - 字面量步长且行参数也是字面量 ⇒ **编译期**判安全区（`compile_time_safe = true`）；
/// - 运行期步长/运行期 `(σ, ω)` ⇒ 允许改写，但 `compile_time_safe = false`
///   （宏层须生成**运行期检查**：`recurrence_safe` 不过 ⇒ 该行**回退 libm**）。
pub fn plan(pattern: &KernelPattern) -> Rewrite {
    // **未挂 `#[fast_trig]` ⇒ 完全不重写**（默认路径行为一字不改）
    if !pattern.fast_trig_opt_in {
        return Rewrite::None;
    }
    if !pattern.lin_in_index {
        return Rewrite::None;
    }
    match pattern.step {
        StepKind::Varies => Rewrite::None,
        StepKind::Literal(dt) => {
            // 字面量步长：**编译期**可判安全区（这里用 0 相位代表"仅步长可知"，
            // 宏层会把字面量 ω 也带进来；未知 ω 时按运行期处理）
            let _ = dt;
            Rewrite::Recurrence {
                compile_time_safe: true,
            }
        }
        StepKind::RuntimeConst => Rewrite::Recurrence {
            compile_time_safe: false,
        },
    }
}

/// **安全区判定（与库内同源）**：`ω·Δt` 与 `ω·Δt·N` 都在安全区内才返回 `true`。
/// 直接复用 [`recurrence_safe`]（同一常量、同一函数）。
pub fn safe(om: f64, dt: f64, n: usize) -> bool {
    recurrence_safe(om, dt, n)
}

/// 安全区常量（**转发库内定义**，供宏层生成运行期检查时引用；不复制数值）。
pub fn safe_zone() -> (f64, f64) {
    (RECURRENCE_MAX_PHASE_STEP, RECURRENCE_MAX_PHASE)
}

/// **生成侧说明（本轮的真实边界）**：本原型**不生成源码文本**，宏层将来在做代码生成时只需：
/// ① 调 [`plan`] 拿判定；② 走 `Rewrite::Recurrence` 时调用**库内已落地的**
/// `laplace_batch_{f32,f64}_recurrence`（它内部就是**同源守卫 + 逐行回退 libm**）。
/// 因此"生成 + 守卫 + 逐行回退"这条链路的**行为**已由库内入口 + 下面的测试覆盖；
/// **尚未落地**的是"宏 crate 里把 AST 降成 `KernelPattern` 并挂 `#[fast_trig]`"（见 `docs/dev.md` §9.5.2）。
pub const GENERATION_VIA_LIBRARY_ENTRY: &str =
    "recurrence 生成走 laplace_batch_*_recurrence（同源守卫 + 逐行回退）";

#[cfg(test)]
mod tests {
    use super::*;

    fn pat(step: StepKind, op: TrigOp, lin: bool) -> KernelPattern {
        KernelPattern {
            base: 0.0,
            step,
            n: 1024,
            op,
            lin_in_index: lin,
            fast_trig_opt_in: true,
        }
    }

    /// **未挂 `#[fast_trig]` ⇒ 完全不重写**（默认路径位精确/libm 不变）。
    #[test]
    fn no_opt_in_means_no_rewrite() {
        let mut p = pat(StepKind::Literal(1.0 / 256.0), TrigOp::Exp, true);
        p.fast_trig_opt_in = false;
        assert_eq!(
            plan(&p),
            Rewrite::None,
            "属性未挂时必须完全不重写（不许静默改写语义）"
        );
        // 反面：同一个 pattern 挂上属性 ⇒ 可改写（证明是"属性"在起作用，不是别的条件）
        p.fast_trig_opt_in = true;
        assert_eq!(
            plan(&p),
            Rewrite::Recurrence {
                compile_time_safe: true
            }
        );
    }

    /// **正例**：等距字面量步长 + 索引线性 ⇒ 可改写，且编译期可判安全区。
    #[test]
    fn positive_literal_isometric() {
        for op in [TrigOp::Exp, TrigOp::Cos, TrigOp::Sin] {
            let r = plan(&pat(StepKind::Literal(1.0 / 256.0), op, true));
            assert_eq!(
                r,
                Rewrite::Recurrence {
                    compile_time_safe: true
                },
                "{op:?} 等距字面量应可改写"
            );
        }
    }

    /// **正例**：运行期常量步长 ⇒ 可改写，但**只能运行期检查**。
    #[test]
    fn positive_runtime_const_needs_runtime_check() {
        assert_eq!(
            plan(&pat(StepKind::RuntimeConst, TrigOp::Exp, true)),
            Rewrite::Recurrence {
                compile_time_safe: false
            }
        );
    }

    /// **负例（不许静默改写）**：
    /// ① 步长非常量/非等距（`Varies`）⇒ 完全不改写；
    /// ② 参数非线性（不是 `a + i·b`）⇒ 完全不改写；
    /// ③ 两者都坏 ⇒ 完全不改写。
    #[test]
    fn negative_non_isometric_or_nonlinear() {
        assert_eq!(
            plan(&pat(StepKind::Varies, TrigOp::Exp, true)),
            Rewrite::None,
            "非等距/非常量步长不许改写"
        );
        assert_eq!(
            plan(&pat(StepKind::Literal(0.01), TrigOp::Cos, false)),
            Rewrite::None,
            "参数非线性（如 t²、t·t、随机项）不许改写"
        );
        assert_eq!(
            plan(&pat(StepKind::Varies, TrigOp::Sin, false)),
            Rewrite::None
        );
    }

    /// **安全区判据与库内同源**：`safe()` 转发 `laplace::recurrence_safe`，
    /// `safe_zone()` 转发库内常量（**同一处定义**，不是各写一份）。
    #[test]
    fn safe_zone_uses_library_constants() {
        let (ds, dp) = safe_zone();
        assert_eq!(ds, RECURRENCE_MAX_PHASE_STEP, "阈值必须来自库内定义");
        assert_eq!(dp, RECURRENCE_MAX_PHASE);
        // 与库内函数逐点一致（含边界）
        for &(om, dt, n) in &[
            (0.0f64, 0.01f64, 64usize),
            (1.0, 0.01, 1024),
            (200.0, 0.01, 512),     // ω·Δt = 2.0 > 1.5 ⇒ 不安全
            (0.5, 0.01, 100_000),   // 相位总量 500 ⇒ **仍在安全区内**
            (0.5, 0.01, 2_000_000), // 相位总量 10000 > 8192 ⇒ 不安全
        ] {
            assert_eq!(safe(om, dt, n), recurrence_safe(om, dt, n));
        }
        assert!(safe(0.5, 0.01, 1024), "安全区内应为 true");
        assert!(
            safe(0.5, 0.01, 100_000),
            "相位总量 500 仍应在安全区（阈值 8192）"
        );
        assert!(!safe(200.0, 0.01, 512), "ω·Δt 超限应为 false");
        assert!(!safe(0.5, 0.01, 2_000_000), "相位总量超限应为 false");
    }

    /// **③ 生成链路的真实行为测试（`N=1024` 特化）**：
    /// - **安全区内的 pattern**：重写路径（库内递推入口）与 libm 路径**容差内一致**；
    /// - **超安全区的 pattern**：**逐行回退 libm** ⇒ 与 libm 路径**逐位一致**（
    ///   这正是"守卫 + 回退"那条链路的验收）。
    #[test]
    fn generated_path_guard_and_fallback() {
        const M: usize = 2;
        const N: usize = 1024;
        const NB: usize = 1;
        let dt = 1.0 / 256.0;
        let f = vec![1.0f64; N * NB];

        // (a) 安全区内
        let sig = [1.0f64, 2.0];
        let om = [0.05f64, 0.10]; // ω·Δt = 1.95e-4 / 3.9e-4 ⇒ 安全
        let p = KernelPattern {
            base: 0.0,
            step: StepKind::Literal(dt),
            n: N,
            op: TrigOp::Exp,
            lin_in_index: true,
            fast_trig_opt_in: true,
        };
        assert!(matches!(plan(&p), Rewrite::Recurrence { .. }));
        assert!(safe(om[0], dt, N) && safe(om[1], dt, N));
        let (rr, ri, all_safe) =
            crate::laplace::laplace_batch_f64_recurrence::<M, N, NB>(&sig, &om, dt, &f);
        assert!(all_safe, "安全区内 all_safe 应为 true");
        let (br, bi) = crate::laplace::laplace_batch_f64::<M, N, NB>(&sig, &om, dt, &f);
        let mut worst = 0.0f64;
        for k in 0..M * NB {
            worst = worst.max((rr[k] - br[k]).abs() / br[k].abs().max(1e-30));
            worst = worst.max((ri[k] - bi[k]).abs() / bi[k].abs().max(1e-30));
        }
        assert!(
            worst < 1e-9,
            "安全区内递推 vs libm 应 <1e-9，实测 {worst:e}"
        );

        // (b) **超安全区**：ω·Δt = 2.0 > RECURRENCE_MAX_PHASE_STEP ⇒ 该行回退 libm
        let om_big = [2.0 / dt, 0.05];
        assert!(!safe(om_big[0], dt, N), "ω·Δt=2.0 应在安全区外");
        let (r2, i2, safe2) =
            crate::laplace::laplace_batch_f64_recurrence::<M, N, NB>(&sig, &om_big, dt, &f);
        assert!(!safe2, "含超区行时 all_safe 应为 false");
        let (b2r, b2i) = crate::laplace::laplace_batch_f64::<M, N, NB>(&sig, &om_big, dt, &f);
        // **按行**：第 0 行超区 ⇒ 逐位回退 libm；第 1 行在安全区 ⇒ 允许舍入级差异
        for b in 0..NB {
            assert_eq!(
                r2[b].to_bits(),
                b2r[b].to_bits(),
                "超区行应逐位回退到 libm（实部）"
            );
            assert_eq!(i2[b].to_bits(), b2i[b].to_bits(), "虚部");
            let e = (r2[NB + b] - b2r[NB + b]).abs() / b2r[NB + b].abs().max(1e-30);
            assert!(e < 1e-9, "安全行只允许舍入级差异，实测 {e:e}");
        }
    }
}
