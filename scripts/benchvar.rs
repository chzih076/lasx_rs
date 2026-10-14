//! 方差测量法的换算探针（入库以便复现；运行产物不进仓库）。
//!
//! 构建：`rustc -O -o /tmp/benchvar scripts/benchvar.rs`
//!
//! 用法：
//!   /tmp/benchvar <discard_rounds> <run1.txt> <run2.txt> ...
//!   /tmp/benchvar render <discard_rounds> <run1.txt> ...   # 直接输出可粘贴的 markdown 行
//!
//!   - 按**行号**对齐各轮输出（`lasx_bench` 的输出顺序确定，逐组跑时组序固定）；
//!   - 每行按出现顺序抽取数值槽位，丢弃前 `discard_rounds` 轮后取**中位数**与**极差**；
//!   - 默认模式输出每行一个记录，便于 `grep` 定位需要的表格行：
//!     `L<行号> <槽位数> <变动槽位数> | <key（去掉数字后的原行）> | 值... | sample=<首轮原行>`
//!     槽位格式：值不变 ⇒ `v`；变动 ⇒ `median(lo-hi)`
//!   - `render` 模式把首轮原行里的每个数值 token 直接替换成 `median(lo-hi)`，
//!     标签与单位原样保留，因此输出就是可直接粘进 `docs/dev.md` 的表格行。
//!
//! **已知边界**：值在轮间跨过格式化的单位边界时（`ns`/`µs`/`ms`、`M/s`/`G/s`），
//! 数值会被当成同一刻度比较 ⇒ 该槽位的极差不可用。默认模式会打印各轮槽位数不一致的行数；
//! 出现 `WARN` 时，受影响的单元格要回原始轮次文件手工核对（task-9 处理了 13 个这样的槽位）。

use std::collections::BTreeSet;
use std::env;
use std::fs;

/// 返回数值 token 的 (值, 起始 char 下标, 结束 char 下标)
fn spans(s: &str) -> Vec<(f64, usize, usize)> {
    let b: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let t: String = b[start..i].iter().collect();
            if let Ok(v) = t.parse::<f64>() {
                out.push((v, start, i));
            }
        } else {
            i += 1;
        }
    }
    out
}

fn nums(s: &str) -> Vec<f64> {
    spans(s).into_iter().map(|(v, _, _)| v).collect()
}

fn key_of(s: &str) -> String {
    let mut out = String::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            out.push('#');
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out.trim_end().to_string()
}

fn fmt(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if a >= 1e7 || a < 1e-4 {
        return format!("{v:.4e}");
    }
    let mut t = format!("{v:.6}");
    if t.contains('.') {
        while t.ends_with('0') {
            t.pop();
        }
        if t.ends_with('.') {
            t.pop();
        }
    }
    t
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: benchvar <discard> <run1> <run2> ...");
        std::process::exit(2);
    }
    let render = args[1] == "render";
    let (discard, first_file) = if render {
        if args.len() < 4 {
            eprintln!("usage: benchvar render <discard> <run1> <run2> ...");
            std::process::exit(2);
        }
        (args[2].parse::<usize>().expect("discard"), 3usize)
    } else {
        if args.len() < 3 {
            eprintln!("usage: benchvar <discard> <run1> <run2> ...");
            std::process::exit(2);
        }
        (args[1].parse::<usize>().expect("discard"), 2usize)
    };
    let files: Vec<Vec<String>> = args[first_file..]
        .iter()
        .map(|p| {
            fs::read_to_string(p)
                .unwrap_or_else(|e| panic!("read {p}: {e}"))
                .lines()
                .map(|l| l.to_string())
                .collect()
        })
        .collect();

    let total = files.len();
    let minlen = files.iter().map(|f| f.len()).min().unwrap();
    let counts: Vec<usize> = files.iter().map(|f| f.len()).collect();
    if !render {
        println!("# runs={total} discard={discard} kept={}", total - discard);
        println!("# line counts: {counts:?}");
        if counts.windows(2).any(|w| w[0] != w[1]) {
            println!("# WARN: 各轮行数不一致 ⇒ 行号对齐不可靠（某组 rc≠0 或输出分支不同）");
        }
        println!("# alignment: by line index, minlen={minlen}");
    }

    let mut inconsistent = 0usize;
    for i in 0..minlen {
        let per_run: Vec<Vec<f64>> = files.iter().map(|f| nums(&f[i])).collect();
        let slot_counts: BTreeSet<usize> = per_run.iter().map(|v| v.len()).collect();
        if slot_counts.len() > 1 {
            inconsistent += 1;
        }
        let kept: Vec<&Vec<f64>> = per_run.iter().skip(discard).collect();
        let maxslots = per_run.iter().map(|v| v.len()).max().unwrap();
        if maxslots == 0 {
            continue;
        }
        let mut slots = Vec::new();
        let mut varying = 0usize;
        for s in 0..maxslots {
            let vals: Vec<f64> = kept
                .iter()
                .filter_map(|v| v.get(s).copied())
                .collect();
            if vals.is_empty() {
                slots.push("—".to_string());
                continue;
            }
            let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let med = median(&vals);
            if (hi - lo).abs() > 1e-12 {
                varying += 1;
                slots.push(format!("{}({}-{})", fmt(med), fmt(lo), fmt(hi)));
            } else {
                slots.push(fmt(med));
            }
        }
        if render {
            // 把原行里的每个数值 token 换成 "中位数(极差)"，标签与单位原样保留
            let src = &files[0][i];
            let chars: Vec<char> = src.chars().collect();
            let sp = spans(src);
            let mut line = String::new();
            let mut pos = 0usize;
            for (k, (_, a, b)) in sp.iter().enumerate() {
                if *a < pos {
                    continue;
                }
                let seg: String = chars[pos..*a].iter().collect();
                line.push_str(&seg);
                line.push_str(slots.get(k).map(|s| s.as_str()).unwrap_or("?"));
                pos = *b;
            }
            let tail: String = chars[pos.min(chars.len())..].iter().collect();
            line.push_str(&tail);
            println!("{line}");
        } else {
            let with_n = slots
                .iter()
                .map(|s| s.clone())
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "L{} {} {} | {} | {} | sample={}",
                i + 1,
                maxslots,
                varying,
                key_of(&files[0][i]),
                with_n,
                files[0][i]
            );
        }
    }
    if !render {
        if inconsistent > 0 {
            println!("# WARN: {inconsistent} 行的槽位数在各轮间不一致（单位切换/条件行），这些行的槽位对齐不可靠");
        } else {
            println!("# OK: 所有行的槽位数在各轮间一致");
        }
    }
}
