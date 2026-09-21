//! 基准分组：一个 `enum` 描述所有可跑的套件，`match` 完成分派与过滤。

/// 可独立选择/过滤的基准组。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// f32 点积。
    Dot,
    /// f32 归约。
    Sum,
    /// `y += a·x`。
    Axpy,
    /// f64 点积。
    DotF64,
    /// int8 量化点积。
    DotI8,
    /// Q4 量化点积。
    DotQ4,
    /// 矩阵乘（f32 + f64）。
    Matmul,
    /// 批量 3 分量模长。
    Norm3,
    /// 批量缩放加。
    Vec3,
    /// 批量 2D 距离。
    Distance2d,
    /// 批量 J2 加速度。
    J2,
    /// 批量弹道步。
    Ballistic,
    /// 批量 RK4 J2 步。
    Rk4,
    /// 纯寄存器 FMA 吞吐微基准。
    FmaPeak,
    /// 多线程扩展性（含线程创建开销）。
    ThreadScaling,
    /// 缓冲区对齐对 LASX 的影响。
    Align,
}

impl Group {
    /// 全部组，按执行顺序排列。
    pub const ALL: &'static [Group] = &[
        Group::Dot,
        Group::Sum,
        Group::Axpy,
        Group::DotF64,
        Group::DotI8,
        Group::DotQ4,
        Group::Matmul,
        Group::Norm3,
        Group::Vec3,
        Group::Distance2d,
        Group::J2,
        Group::Ballistic,
        Group::Rk4,
        Group::FmaPeak,
        Group::ThreadScaling,
        Group::Align,
    ];

    /// 组名，同时用作命令行过滤子串。
    pub fn name(self) -> &'static str {
        match self {
            Group::Dot => "dot",
            Group::Sum => "sum",
            Group::Axpy => "axpy",
            Group::DotF64 => "dot_f64",
            Group::DotI8 => "dot_i8",
            Group::DotQ4 => "dot_q4",
            Group::Matmul => "matmul",
            Group::Norm3 => "norm3",
            Group::Vec3 => "vec3",
            Group::Distance2d => "distance2d",
            Group::J2 => "j2",
            Group::Ballistic => "ballistic",
            Group::Rk4 => "rk4",
            Group::FmaPeak => "fma",
            Group::ThreadScaling => "mt",
            Group::Align => "align",
        }
    }

    /// 该组是否输出到结果表格（微基准自行打印，不写表）。
    pub fn is_tabular(self) -> bool {
        !matches!(self, Group::FmaPeak | Group::ThreadScaling | Group::Align)
    }

    /// 是否被命令行过滤串选中（空串选中全部）。
    pub fn matches(self, filter: &str) -> bool {
        filter.is_empty() || self.name().contains(filter)
    }
}
