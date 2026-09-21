//! 结果表：一行一个（内核, 规模）测量，渲染成 Markdown 表格。

use crate::timing::{fmt_t, fmt_thr};
use std::time::Duration;

pub struct Row {
    kernel: &'static str,
    n: String,
    work: f64,
    unit: &'static str,
    lasx: Duration,
    lsx: Option<Duration>,
    scalar: Duration,
}
#[allow(clippy::too_many_arguments)]
pub fn row3(
    kernel: &'static str,
    n: String,
    work: f64,
    unit: &'static str,
    lasx: Duration,
    lsx: Option<Duration>,
    scalar: Duration,
    rows: &mut Vec<Row>,
) {
    rows.push(Row {
        kernel,
        n,
        work,
        unit,
        lasx,
        lsx,
        scalar,
    });
}
pub fn report(rows: &[Row]) {
    println!("| 内核 | n | LASX | 强制 LSX | 标量 | LASX/标量 | LASX/LSX | LASX 吞吐 |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in rows {
        let sx = r.scalar.as_secs_f64() / r.lasx.as_secs_f64();
        let lx = r.lsx.map(|d| d.as_secs_f64() / r.lasx.as_secs_f64());
        let thr = r.work / r.lasx.as_secs_f64();
        println!(
            "| {} | {} | {} | {} | {} | {:.2}× | {} | {} {} |",
            r.kernel,
            r.n,
            fmt_t(r.lasx),
            r.lsx.map(fmt_t).unwrap_or_else(|| "—".into()),
            fmt_t(r.scalar),
            sx,
            lx.map(|v| format!("{v:.2}×")).unwrap_or_else(|| "—".into()),
            fmt_thr(thr),
            r.unit,
        );
    }
}
