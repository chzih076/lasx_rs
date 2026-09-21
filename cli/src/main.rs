//! lasx_rs 性能基准 CLI。
//!
//! 用法：
//!
//! ```text
//! cargo run -p lasx_bench --release            # 跑全部
//! cargo run -p lasx_bench --release -- matmul  # 只跑组名含 matmul 的套件
//! cargo run -p lasx_bench --release -- i8      # 只跑 int8 点积
//! ```
//!
//! 计时口径：预热 2 次后自适应重复至 ≈25 ms/样品，5 次取样取**中位数**，
//! 每个内核在同一个线程内运行（强制 LSX 的钩子是线程级的）。
#![feature(stdarch_loongarch)]

mod data;
mod group;
mod report;
mod scalar_ref;
mod suites;
mod timing;

use group::Group;
use report::report;

pub fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Model Name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let selected: Vec<Group> = Group::ALL
        .iter()
        .copied()
        .filter(|g| g.matches(&filter))
        .collect();

    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    println!("# lasx_rs 性能基准");
    println!();
    println!("- 主机：{}（LoongArch64，LASX）", cpu_model());
    println!("- 可用并行度：{threads}");
    println!("- 计时：预热后自适应重复至 ~25 ms/样品，5 次取中位数");
    println!("- 标量列：LLVM 在 +lasx 下会自动向量化其中大部分，见 scalar_ref 模块说明");
    if filter.is_empty() {
        println!("- 用法：`cargo run -p lasx_bench --release [-- <过滤子串>]`");
    } else {
        println!("- 过滤：只跑组名含 `{filter}` 的套件");
    }
    println!();

    if selected.is_empty() {
        let names: Vec<&str> = Group::ALL.iter().map(|g| g.name()).collect();
        println!(
            "（过滤 `{filter}` 未匹配任何套件；可用：{}）",
            names.join(", ")
        );
        return;
    }

    let mut rows = Vec::new();
    // 先跑"往表格里写行"的套件并渲染表格，再跑自行打印的微基准，
    // 保证输出顺序稳定（表格在前、微基准在后）。
    for g in selected.iter().copied().filter(|g| g.is_tabular()) {
        dispatch(g, &mut rows);
    }
    if !rows.is_empty() {
        report(&rows);
    }
    for g in selected.iter().copied().filter(|g| !g.is_tabular()) {
        dispatch(g, &mut rows);
    }
}

/// 单个套件的分派：`match` 是 [`Group`] 与各套件实现之间的唯一映射点。
fn dispatch(group: Group, rows: &mut Vec<report::Row>) {
    match group {
        Group::Dot => suites::reduce::dot(rows),
        Group::Sum => suites::reduce::sum(rows),
        Group::Axpy => suites::reduce::axpy(rows),
        Group::DotF64 => suites::reduce::dot_f64(rows),
        Group::DotI8 => suites::quant::dot_i8(rows),
        Group::DotQ4 => suites::quant::dot_q4(rows),
        Group::Matmul => suites::matmul::matmul(rows),
        Group::Norm3 => suites::batch::norm3(rows),
        Group::Vec3 => suites::batch::vec3(rows),
        Group::Distance2d => suites::batch::distance2d(rows),
        Group::J2 => suites::physics::j2(rows),
        Group::Ballistic => suites::physics::ballistic(rows),
        Group::Rk4 => suites::physics::rk4(rows),
        Group::FmaPeak => suites::micro::fma_peak(),
        Group::ThreadScaling => suites::micro::thread_scaling(),
        Group::Align => suites::align::align(),
        Group::DispatchOverhead => suites::micro::dispatch_overhead(),
        Group::Scenario => suites::scenario::run_all(),
    }
}
