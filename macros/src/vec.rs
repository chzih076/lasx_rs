//! `dot!` 与 `gemv!`：`matmul!` 的**一维扩展**（向量点积、矩阵-向量乘）。
//!
//! ```text
//! dot!(x[K] * y[K]);                          // 标量结果（两个 1 下标操作数）
//! dot!(a_f16[K] * x[K]);                       // f16 权重 · f32 向量，两个方向都认
//! gemv!(w[N, K] * x[K]);                       // 分配输出：VecBuf<f32, N>
//! gemv!(y[N] = w[N, K] * x[K]);                // 写进已有 VecBuf<f32, N>
//! ```
//!
//! 与 `matmul!` 共享同一条核心规则（下标首字符分派 const/运行期）**和同一套诊断样式**，
//! 但一维形态在 v1 里**只支持编译期长度**：小写下标会得到一条"暂不支持"的诊断，而不是
//! 静默生成无法编译的代码。
//!
//! # 为什么 `gemv!` 只认一种读法
//!
//! `lasx_gemv_f16` 的权重是**行主序 `[输出, 收缩]`**：`y[r] = Σ_k W[r, k] · x[k]`。所以
//! `w[N, K] * x[K]` 是唯一与内核布局一致的写法。反过来写（`x[K] * w[N, K]`）需要
//! `[K, N]` 布局（列主序打包），本库的 gemv 不提供 —— 与其让它落成一条难懂的 rustc 类型
//! 错误，不如在宏里报错并说明该怎么写。
//!
//! # 结构：规则（纯函数、可单测）+ 翻译（Span 映射）
//!
//! proc-macro 的类型（`Span`/`Ident`）在**测试里用不了**（"procedural macro API is used
//! outside of a procedural macro"），所以这里分两层：
//!
//! - [`check_dot`] / [`check_gemv`]：只吃**下标名字**，返回 [`Violation`]（纯数据）——
//!   单元测试直接打这两层；
//! - `diagnose_*`：把 [`Violation`] 按下标序号映射回用户 token 的 `Span`，产出 [`Diagnostic`]
//!   （**穷尽 `match`，不留 `_` 分支**：新增违规种类时编译器会指着这里）。
//!
//! # 诊断一览
//!
//! | 情况 | 消息要点 |
//! |---|---|
//! | `dot!` 里写了输出（`s = …`） | 点积结果是标量，没有"输出操作数" |
//! | `dot!` 的操作数是 2 下标 | 形状不对：点积两侧都是向量（矩阵请用 `matmul!`/`gemv!`） |
//! | `dot!` 两侧下标不同名 | 双锚定：收缩维必须同名 |
//! | `gemv!` 左侧不是矩阵 | 左侧必须是权重 `w[N, K]`；若右侧是矩阵则提示权重在左 |
//! | `gemv!` 右侧是矩阵 | 用 `matmul!` |
//! | `gemv!` 收缩维/输出维下标不同名 | 双锚定 |
//! | 任一处小写下标 | v1 的向量形态只支持编译期长度 |
//! | 尾巴上还有 token（`+`、多余操作数） | 公式到乘积就结束了 |

use proc_macro::{Delimiter, Ident, Punct, Spacing, Span, TokenStream, TokenTree};

use crate::{amp, const_arg, group, id, method, p, Diagnostic, Kind};

/* ==================== 纯规则层（可单测） ==================== */

/// 违规种类：只带**下标序号**，不带 `Span`（Span 由翻译层补）。
#[derive(Debug, PartialEq, Eq)]
enum Violation {
    /// `dot!` 的操作数是矩阵（2 下标）。`li`/`ri` 指出是哪一个。
    DotNotVector { left: bool },
    /// `dot!` 两侧收缩维不同名。
    ContractionMismatch { lhs: String, rhs: String },
    /// `gemv!` 左侧不是矩阵（1 下标）。`rhs_is_mat` 用于给出"权重在左"的正确写法。
    GemvLeftNotMatrix { rhs_is_mat: bool },
    /// `gemv!` 右侧是矩阵（2 下标）。
    GemvRightIsMatrix,
    /// `gemv!` 输出维与权重行维不同名。
    OutputRowMismatch { out: String, lhs: String },
}

/// `dot!` 的形状规则：两侧都必须是 1 下标，且收缩维同名。
fn check_dot(lhs: &[String], rhs: &[String]) -> Result<(), Violation> {
    if lhs.len() != 1 {
        return Err(Violation::DotNotVector { left: true });
    }
    if rhs.len() != 1 {
        return Err(Violation::DotNotVector { left: false });
    }
    if lhs[0] != rhs[0] {
        return Err(Violation::ContractionMismatch {
            lhs: lhs[0].clone(),
            rhs: rhs[0].clone(),
        });
    }
    Ok(())
}

/// `gemv!` 的形状规则：左矩阵（`[行, 收缩]`）、右向量（`[收缩]`），输出维（若写了）与行同名。
fn check_gemv(lhs: &[String], rhs: &[String], out: Option<&[String]>) -> Result<(), Violation> {
    if lhs.len() != 2 {
        return Err(Violation::GemvLeftNotMatrix {
            rhs_is_mat: rhs.len() == 2,
        });
    }
    if rhs.len() != 1 {
        return Err(Violation::GemvRightIsMatrix);
    }
    if lhs[1] != rhs[0] {
        return Err(Violation::ContractionMismatch {
            lhs: lhs[1].clone(),
            rhs: rhs[0].clone(),
        });
    }
    if let Some(o) = out {
        if o.len() != 1 {
            return Err(Violation::DotNotVector { left: false });
        }
        if o[0] != lhs[0] {
            return Err(Violation::OutputRowMismatch {
                out: o[0].clone(),
                lhs: lhs[0].clone(),
            });
        }
    }
    Ok(())
}

/* ==================== 解析层（proc-macro 类型） ==================== */

/// 一个带 1 或 2 个下标的操作数：`x[K]`（向量）或 `w[N, K]`（矩阵）。
struct VecOperand {
    name: Ident,
    idx: Vec<Ident>,
}

impl VecOperand {
    /// 取下标**名字**（进纯规则层）。
    fn dims(&self) -> Vec<String> {
        self.idx.iter().map(|d| d.to_string()).collect()
    }

    /// 第 `i` 个下标的 Span（报错锚点）。
    fn span(&self, i: usize) -> Span {
        self.idx[i].span()
    }

    /// 该操作数所有下标都必须是编译期 const（v1 的向量形态限制）。
    fn require_const(&self) -> Result<(), Diagnostic> {
        for d in &self.idx {
            let kind = Kind::of(&d.to_string()).map_err(|msg| Diagnostic::new(msg, d.span()))?;
            if kind == Kind::Runtime {
                return Err(Diagnostic::new(
                    format!("`{}` 的下标 `{d}` 是运行期值", self.name),
                    d.span(),
                )
                .with_note(
                    "`dot!` / `gemv!` 目前只支持编译期长度（`VecRef`/`F16Vec`/`F16Mat` 的 `N`/`K` 是常量）；\
                     运行期长度请直接用 `api::dot_f16` / `parallel::gemv_f16`",
                ));
            }
        }
        Ok(())
    }
}

/// 解析一个操作数：`ident '[' ident (',' ident)? ']'`。
fn parse_operand(tokens: &[TokenTree], at: usize) -> Result<(VecOperand, usize), Diagnostic> {
    let Some(found) = tokens.get(at) else {
        return Err(Diagnostic::new(
            "公式在这里就结束了，缺少操作数",
            Span::call_site(),
        ));
    };
    let TokenTree::Ident(name) = found else {
        return Err(Diagnostic::new(
            "这里应该是操作数（形如 `x[K]` 或 `w[N, K]`）",
            found.span(),
        ));
    };
    let name = name.clone();
    let name_span = name.span();

    let Some(TokenTree::Group(arg)) = tokens.get(at + 1) else {
        return Err(Diagnostic::new(
            format!("`{name}` 后面缺少下标：向量写 `{name}[K]`，矩阵写 `{name}[N, K]`"),
            name_span,
        ));
    };
    if arg.delimiter() != Delimiter::Bracket {
        return Err(Diagnostic::new(
            format!("`{name}` 的下标必须写在方括号里：向量 `{name}[K]`、矩阵 `{name}[N, K]`"),
            arg.span(),
        ));
    }

    let parts: Vec<TokenTree> = arg.stream().into_iter().collect();
    let idx = match parts.as_slice() {
        [TokenTree::Ident(i)] => vec![i.clone()],
        [TokenTree::Ident(i), TokenTree::Punct(c), TokenTree::Ident(j)] if c.as_char() == ',' => {
            if i.to_string() == j.to_string() {
                return Err(Diagnostic::new(
                    format!("同一次操作数的两个下标不能同名（`{name}[{i}, {i}]`）"),
                    i.span(),
                )
                .with_second(j.span())
                .with_note("两个下标必须分别表示行与列"));
            }
            vec![i.clone(), j.clone()]
        }
        _ => {
            return Err(Diagnostic::new(
                format!("`{name}` 的下标必须是 1 个或 2 个标识符：`{name}[K]` / `{name}[N, K]`"),
                arg.span(),
            ))
        }
    };
    Ok((VecOperand { name, idx }, at + 2))
}

/// 解析 `a[..] * b[..]`，返回两侧操作数与"下一个待读 token 的位置"。
fn parse_product(
    tokens: &[TokenTree],
    at: usize,
) -> Result<(VecOperand, VecOperand, usize), Diagnostic> {
    let (lhs, at) = parse_operand(tokens, at)?;
    match tokens.get(at) {
        Some(TokenTree::Punct(star)) if star.as_char() == '*' => {}
        Some(other) => {
            return Err(Diagnostic::new(
                "这里应该是 `*`（公式只接受单项乘积）",
                other.span(),
            ))
        }
        None => {
            return Err(Diagnostic::new(
                "公式在这里就结束了，缺少 `* 操作数`",
                Span::call_site(),
            ))
        }
    }
    let (rhs, at) = parse_operand(tokens, at + 1)?;
    Ok((lhs, rhs, at))
}

/// 乘积之后不允许再有 token（`+`、第二个乘积……）。
fn expect_end(tokens: &[TokenTree], at: usize) -> Result<(), Diagnostic> {
    let Some(tail) = tokens.get(at) else {
        return Ok(());
    };
    if matches!(tail, TokenTree::Punct(c) if c.as_char() == '+') {
        return Err(Diagnostic::new(
            "v1 不支持 `+`：多操作数/累加还没实现（先算 `dot!(…)` 再自己相加）",
            tail.span(),
        )
        .with_note("多操作数、`alpha` 缩放都在 DSL 的下一步计划里，尚未启用"));
    }
    Err(Diagnostic::new("公式到乘积就结束了", tail.span()))
}

/* ==================== 翻译层：Violation → Diagnostic（穷尽 match） ==================== */

fn diagnose_dot(v: Violation, lhs: &VecOperand, rhs: &VecOperand) -> Diagnostic {
    match v {
        Violation::DotNotVector { left } => {
            let (op, at) = if left {
                (lhs, lhs.idx.len().min(1))
            } else {
                (rhs, rhs.idx.len().min(1))
            };
            Diagnostic::new(
                format!(
                    "`dot!` 的两个操作数都是向量（1 下标），`{}` 写了 {} 个下标",
                    op.name,
                    op.idx.len()
                ),
                op.span(at),
            )
            .with_note("矩阵乘用 `matmul!`，矩阵-向量用 `gemv!`")
        }
        Violation::ContractionMismatch { lhs: l, rhs: r } => Diagnostic::new(
            format!("点积的收缩维下标不一致：左侧写 {l}，右侧写 {r}"),
            lhs.span(0),
        )
        .with_second(rhs.span(0))
        .with_note("点积要求两侧等长，请把下标写成同一个名字"),
        // 下面两类只可能从 `check_gemv` 出来；`dot!` 路径上不可达，但 match 不省分支。
        Violation::GemvLeftNotMatrix { .. } | Violation::GemvRightIsMatrix => Diagnostic::new(
            "`dot!` 的形状检查出现内部不一致（请报 issue）",
            lhs.name.span(),
        ),
        Violation::OutputRowMismatch { out, lhs: l } => Diagnostic::new(
            format!("输出维下标不一致：输出写 {out}，权重行写 {l}"),
            lhs.name.span(),
        ),
    }
}

fn diagnose_gemv(
    v: Violation,
    lhs: &VecOperand,
    rhs: &VecOperand,
    out: Option<&VecOperand>,
) -> Diagnostic {
    match v {
        Violation::GemvLeftNotMatrix { rhs_is_mat } => {
            let note = if rhs_is_mat {
                format!(
                    "本库 gemv 的权重是行主序 `[输出, 收缩]`，权重必须在左：`w[{}, {}] * {}[{}]`",
                    rhs.dims()[1],
                    rhs.dims()[0],
                    lhs.name,
                    rhs.dims()[0]
                )
            } else {
                "两边都是向量：点积请用 `dot!`".to_string()
            };
            Diagnostic::new(
                format!(
                    "`gemv!` 需要左侧是权重矩阵（2 下标），`{}` 只写了 1 个下标",
                    lhs.name
                ),
                lhs.span(0),
            )
            .with_note(note)
        }
        Violation::GemvRightIsMatrix => Diagnostic::new(
            format!(
                "`gemv!` 的右侧是向量（1 下标），`{}` 写了 2 个下标",
                rhs.name
            ),
            rhs.span(1),
        )
        .with_note("两个矩阵相乘用 `matmul!`"),
        Violation::ContractionMismatch { lhs: l, rhs: r } => Diagnostic::new(
            format!("收缩维下标不一致：权重写 {l}，向量写 {r}"),
            lhs.span(1),
        )
        .with_second(rhs.span(0))
        .with_note("权重每行的长度必须等于向量长度，请写成同一个下标"),
        Violation::OutputRowMismatch { out: o, lhs: l } => {
            let (span, second) = match out {
                Some(op) => (op.span(0), lhs.span(0)),
                // 没有输出操作数时这条不可能到达；锚在权重行上，别丢信息。
                None => (lhs.span(0), lhs.span(0)),
            };
            Diagnostic::new(format!("输出维下标不一致：输出写 {o}，权重行写 {l}"), span)
                .with_second(second)
                .with_note("输出长度必须等于权重行数")
        }
        // `dot!` 专有的两类：`gemv!` 路径上不可达，但 match 不省分支。
        Violation::DotNotVector { left } => {
            let op = if left { lhs } else { rhs };
            Diagnostic::new(
                format!(
                    "`gemv!` 的输出操作数必须是向量（1 下标），`{}` 不是",
                    op.name
                ),
                op.span(0),
            )
        }
    }
}

/* ==================== 代码生成 ==================== */

/// `dot!(a[K] * b[K])` → `a.dot_with(&b)`。
///
/// 结果类型由 [`lasx_rs::shape::vec::Dot`] 的 impl 决定（`f32`/`f64`），**不需要**类型标注：
/// 两边的长度绑在同一个常量 `N` 上，dtype 不匹配时 rustc 用 `on_unimplemented` 给提示。
pub(crate) fn expand_dot(input: TokenStream) -> TokenStream {
    match dot(input.into_iter().collect()) {
        Ok(ts) => ts,
        Err(d) => d.emit(),
    }
}

fn dot(tokens: Vec<TokenTree>) -> Result<TokenStream, Diagnostic> {
    // `dot!` 没有输出操作数：`=` 出现在这里就是写法错误（点积结果是标量）。
    if let Some(eq) = tokens
        .iter()
        .find(|t| matches!(t, TokenTree::Punct(c) if c.as_char() == '='))
    {
        return Err(Diagnostic::new(
            "`dot!` 的结果是标量，没有输出操作数（去掉 `= …`，直接 `let s = dot!(a[K] * b[K]);`）",
            eq.span(),
        ));
    }
    let (lhs, rhs, at) = parse_product(&tokens, 0)?;
    expect_end(&tokens, at)?;
    lhs.require_const()?;
    rhs.require_const()?;
    check_dot(&lhs.dims(), &rhs.dims()).map_err(|v| diagnose_dot(v, &lhs, &rhs))?;

    // `lasx_rs::shape::vec::dot_of((&a, &b))`：一个泛型函数，把 `Pair: DotPair` 的求解交给
    // trait 系统 —— 不支持的 dtype 组合才会走到 `DotPair` 的 `on_unimplemented` 提示；
    // 同时它是绝对路径，用户不需要 `use` 任何 trait。**整对做成元组**：两个独立实参时
    // rustc 会拿第二个去凑第一个类型的唯一 impl，报成"实参类型不符"。
    let mut tuple = TokenStream::new();
    tuple.extend(amp(&lhs.name));
    tuple.extend([p(','), p('&'), TokenTree::Ident(rhs.name.clone())]);
    let mut args = TokenStream::new();
    args.extend([group(Delimiter::Parenthesis, tuple)]);
    let mut call = path_of(&["shape", "vec", "dot_of"]);
    call.extend([group(Delimiter::Parenthesis, args)]);
    Ok(stream_of(group(Delimiter::Brace, call)))
}

/// `gemv!(y[N] = w[N, K] * x[K])` / `gemv!(w[N, K] * x[K])`。
pub(crate) fn expand_gemv(input: TokenStream) -> TokenStream {
    match gemv(input.into_iter().collect()) {
        Ok(ts) => ts,
        Err(d) => d.emit(),
    }
}

fn gemv(tokens: Vec<TokenTree>) -> Result<TokenStream, Diagnostic> {
    // 可选的 `out[I] =` 前缀：先解析一个操作数，看后面是不是 `=`。
    let (first, at) = parse_operand(&tokens, 0)?;
    let (out, at) = match tokens.get(at) {
        Some(TokenTree::Punct(eq)) if eq.as_char() == '=' => (Some(first), at + 1),
        _ => (None, 0),
    };
    let (lhs, rhs, at) = parse_product(&tokens, at)?;
    expect_end(&tokens, at)?;
    if let Some(o) = &out {
        o.require_const()?;
    }
    lhs.require_const()?;
    rhs.require_const()?;
    check_gemv(
        &lhs.dims(),
        &rhs.dims(),
        out.as_ref().map(|o| o.dims()).as_deref(),
    )
    .map_err(|v| diagnose_gemv(v, &lhs, &rhs, out.as_ref()))?;

    let n = const_arg(&lhs.idx[0]);
    let k = const_arg(&lhs.idx[1]);

    let mut body = TokenStream::new();
    // let _: &F16Mat<'_, {N}, {K}> = &w;   let _: &VecRef<'_, _, {K}> = &x;
    // （有输出时再加 let _: &mut VecBuf<_, {N}> = &mut y;）
    body.extend(annotate_ref_elemless(
        "F16Mat",
        vec![n.clone(), k.clone()],
        &lhs.name,
    ));
    body.extend(annotate_ref_elem("VecRef", vec![k.clone()], &rhs.name));
    let mut args = TokenStream::new();
    args.extend(amp(&rhs.name));
    if let Some(o) = &out {
        body.extend(annotate_mut("VecBuf", vec![n.clone()], &o.name));
        args.extend([p(','), p('&'), id("mut"), TokenTree::Ident(o.name.clone())]);
        body.extend(method(&lhs.name, "gemv_into", args));
        body.extend([p(';')]);
    } else {
        body.extend(method(&lhs.name, "gemv", args));
    }
    Ok(stream_of(group(Delimiter::Brace, body)))
}

/// 把单个 `TokenTree` 包成 `TokenStream`（`group(...)` 返回的是树，入口要流）。
fn stream_of(tt: TokenTree) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([tt]);
    ts
}

/// `lasx_rs::seg1::seg2…`（绝对路径：生成代码不依赖用户 `use` 了什么）。
fn path_of(segments: &[&str]) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([id("lasx_rs")]);
    for s in segments {
        ts.extend(colon2());
        ts.extend([id(s)]);
    }
    ts
}

/// `::`：第一个 `:` 必须是 `Joint`，否则 rustc 读到的是两个独立冒号（踩过：报
/// `expected expression, found ':'`）。
fn colon2() -> [TokenTree; 2] {
    [
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
    ]
}

/* ------------------------------ 类型标注 ------------------------------ */

/// `let _: &Ty<'_, _, {C…}> = &x;`（元素类型用 `_` 让 rustc 推；用户 token 的 Span 保留）。
fn annotate_ref_elem(ty: &str, consts: Vec<TokenTree>, value: &Ident) -> TokenStream {
    annotate(ty, true, true, consts, value)
}

/// `let _: &Ty<'_, {C…}> = &x;`（类型没有元素类型参数，如 `F16Mat`）。
fn annotate_ref_elemless(ty: &str, consts: Vec<TokenTree>, value: &Ident) -> TokenStream {
    annotate(ty, true, false, consts, value)
}

/// `let _: &mut Ty<_, {C…}> = &mut x;`（拥有型输出，如 `VecBuf`）。
fn annotate_mut(ty: &str, consts: Vec<TokenTree>, value: &Ident) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([
        id("let"),
        id("_"),
        p(':'),
        p('&'),
        id("mut"),
        type_ts(ty, false, true, consts),
        p('='),
        p('&'),
        id("mut"),
        TokenTree::Ident(value.clone()),
        p(';'),
    ]);
    ts
}

/// `let _: &Ty<...> = &x;`
fn annotate(
    ty: &str,
    view: bool,
    elem: bool,
    consts: Vec<TokenTree>,
    value: &Ident,
) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([
        id("let"),
        id("_"),
        p(':'),
        p('&'),
        type_ts(ty, view, elem, consts),
        p('='),
        p('&'),
        TokenTree::Ident(value.clone()),
        p(';'),
    ]);
    ts
}

/// `Ty<'_, _, {C…}>`：`view` 决定要不要生命周期参数，`elem` 决定要不要元素类型位。
fn type_ts(ty: &str, view: bool, elem: bool, consts: Vec<TokenTree>) -> TokenTree {
    let mut generics = TokenStream::new();
    let mut first = true;
    let push_sep = |ts: &mut TokenStream, first: &mut bool| {
        if !*first {
            ts.extend([p(',')]);
        }
        *first = false;
    };
    if view {
        push_sep(&mut generics, &mut first);
        generics.extend(lifetime_underscore());
    }
    if elem {
        push_sep(&mut generics, &mut first);
        generics.extend([id("_")]);
    }
    for c in consts {
        push_sep(&mut generics, &mut first);
        generics.extend([c]);
    }
    let mut inner = TokenStream::new();
    inner.extend(path_of(&["shape", ty]));
    inner.extend([p('<')]);
    inner.extend(generics);
    inner.extend([p('>')]);
    group(Delimiter::None, inner)
}

/// `'_`（生命周期参数，`Span::call_site()` 即可——它不承载用户语义）。
fn lifetime_underscore() -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([
        TokenTree::Punct(Punct::new('\'', Spacing::Joint)),
        TokenTree::Ident(Ident::new("_", Span::call_site())),
    ]);
    ts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_string()).collect()
    }

    /// `dot!` 的规则：两侧各 1 下标、收缩维同名。
    #[test]
    fn dot_rules() {
        assert_eq!(check_dot(&names(&["K"]), &names(&["K"])), Ok(()));
        assert_eq!(
            check_dot(&names(&["K"]), &names(&["Q"])),
            Err(Violation::ContractionMismatch {
                lhs: "K".into(),
                rhs: "Q".into()
            })
        );
        assert_eq!(
            check_dot(&names(&["M", "K"]), &names(&["K"])),
            Err(Violation::DotNotVector { left: true })
        );
        assert_eq!(
            check_dot(&names(&["K"]), &names(&["N", "K"])),
            Err(Violation::DotNotVector { left: false })
        );
    }

    /// `gemv!` 的规则：左矩阵 `[行, 收缩]`、右向量 `[收缩]`、输出 `[行]`。
    #[test]
    fn gemv_rules() {
        assert_eq!(
            check_gemv(&names(&["N", "K"]), &names(&["K"]), None),
            Ok(())
        );
        assert_eq!(
            check_gemv(&names(&["N", "K"]), &names(&["K"]), Some(&names(&["N"]))),
            Ok(())
        );
        // 权重在右 / 两个向量 / 两个矩阵
        assert_eq!(
            check_gemv(&names(&["K"]), &names(&["N", "K"]), None),
            Err(Violation::GemvLeftNotMatrix { rhs_is_mat: true })
        );
        assert_eq!(
            check_gemv(&names(&["K"]), &names(&["K"]), None),
            Err(Violation::GemvLeftNotMatrix { rhs_is_mat: false })
        );
        assert_eq!(
            check_gemv(&names(&["N", "K"]), &names(&["N", "Q"]), None),
            Err(Violation::GemvRightIsMatrix)
        );
        // 收缩维/输出维不同名
        assert_eq!(
            check_gemv(&names(&["N", "K"]), &names(&["Q"]), None),
            Err(Violation::ContractionMismatch {
                lhs: "K".into(),
                rhs: "Q".into()
            })
        );
        assert_eq!(
            check_gemv(&names(&["N", "K"]), &names(&["K"]), Some(&names(&["M"]))),
            Err(Violation::OutputRowMismatch {
                out: "M".into(),
                lhs: "N".into()
            })
        );
    }
}
