//! **中性内核类型**：把原先来自 `simd_dsl` 契约的值枚举/结构**在 lasx 侧本地定义**。
//!
//! 背景：`lasx_rs` 已**纯化**为纯 ISA 内核库，依赖方向翻转为 **DSL 仓 → lasx**；
//! 本模块只描述"内核要吃哪些运算种类"，不含任何外部 crate 类型。
//! 外部适配器（DSL 仓）负责把自己的契约类型映射到这里的 `*Kind`/`RowFuseSpec`。
//!
//! 等价类口径（与旧 `simd_dsl` 契约逐条一致）：`reduce_rows_map` 的**受支持组合**与
//! `simd_dsl` 标量参照逐位；**不支持的组合**由适配器回退到 DSL 侧标量实现。

/// 行统计量（`RowFuseSpec::reduce`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowReduceKind {
    /// 行和：`Σ_c x[r,c]`
    Sum,
    /// 行平方和：`Σ_c x[r,c]²`
    SumSq,
}

/// 每行统计量的收尾（`RowFuseSpec::epilogue`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowEpilogueKind {
    /// 不改变统计量（`k = stat`）。
    Raw,
    /// 均值：`k = stat / cols`。
    Mean,
    /// 均值加下限：`k = stat / cols + eps`（注意乘法次序与旧契约一致：`stat * inv_cols + eps`）。
    MeanPlusEps,
    /// `k = sqrt(stat/cols + eps)`（RMSNorm 分母）。
    SqrtMeanPlusEps,
    /// `k = 1 / sqrt(stat/cols + eps)`（LayerNorm 分母）。
    InvSqrtMeanPlusEps,
    /// `k = 1 / stat`。
    Inv,
}

/// 行内逐元素组合（`RowFuseSpec::map`）：`out = x op k`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowMapKind {
    /// `x + k`
    Add,
    /// `x − k`
    Sub,
    /// `x · k`
    Mul,
    /// `x / k`
    Div,
}

/// 窄三元组融合规格：行统计（`reduce`）→ 标量收尾（`epilogue`）→ 行内逐元素 map（`map`）。
///
/// 等价于旧 `simd_dsl::RowFuse` 的四个字段（`reduce`/`epilogue`/`map`/`eps`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowFuseSpec {
    /// 行统计量怎么算。
    pub reduce: RowReduceKind,
    /// 每行标量怎么收尾（`Raw` ⇒ `k = stat`）。
    pub epilogue: RowEpilogueKind,
    /// 逐元素怎么与原值组合。
    pub map: RowMapKind,
    /// `epilogue` 里的下限保护（不用 `eps` 的变体忽略它）。
    pub eps: f32,
}

/// **"边 map 边累加"的映射口径**（`map_rows_sum` 用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceMapKind {
    /// `out = exp(x)`。
    Exp,
    /// `out = exp(x − shift[r])`。
    ExpShiftedRow,
}
