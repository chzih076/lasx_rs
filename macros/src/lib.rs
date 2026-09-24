//! `lasx_rs` 的公式 DSL：把矩阵乘写成数学式子，形状编译期对齐。
//!
//! ```text
//! matmul!(y[M, N] = x[M, K] * w[K, N]);     // 全静态：M/K/N 是 const
//! matmul!(y[m, N] = x[m, K] * w[K, N]);     // DYN：m 是运行期行数，K/N 仍是 const
//! ```
//!
//! 这个 crate 是**内部实现**（`publish = false`），用户只看到 `lasx_rs::matmul!`：
//! `lasx_rs` 依赖它并 `pub use` 出来，所以依赖树里仍然只有一个 `lasx_rs`。
//!
//! # 为什么不引 `syn`/`quote`
//!
//! **零第三方依赖**是 `lasx_rs` 的硬主张，所以这里手写 `TokenStream` 解析。公式的文法很小
//! （`操作数 [下标, 下标]`），自己走一遍 token 树比拉两个 proc-macro 生态 crate 更划算，
//! 代价是解析与报错都要自己写——报错质量恰好是这套 DSL 的主要产出之一。
//!
//! # 核心规则：**按 ident 首字符分派**
//!
//! 这是上 proc-macro 的第一条理由（`docs/dev.md §19.3`）：`macro_rules!` 拿不到 token 的
//! 文本，读不到首字符，所以做不到这件事。规则：
//!
//! | 下标写法 | 判定 | 生成什么 |
//! |---|---|---|
//! | 大写开头（`M`、`N`、`K`） | 编译期维度（const） | 类型标注 `let _: &Mat<_, {M}, {K}> = &x;` |
//! | 小写或 `_` 开头（`m`、`_m`、`m2`） | 运行期值（`usize`） | 宏**自己绑定**：`let m = x.rows();` + 断言 |
//!
//! 于是**一个宏同时覆盖静态与 DYN 两档**，不需要 `matmul!` / `matmul_dyn!` 两个宏。
//!
//! # 文法与两种读法
//!
//! ```text
//! 公式   := 操作数 '=' 操作数 '*' 操作数
//!         | 操作数 '*' 操作数
//! 操作数 := ident '[' ident ',' ident ']'
//! ```
//!
//! 两个操作数**按"共享下标出现在哪一侧"判角色**，两种读法都认：
//!
//! | 写法 | 判定 | 角色 |
//! |---|---|---|
//! | `x[M,K] * w[K,N]` | 左.j == 右.i | 左是输入、右是权重（`A·B`） |
//! | `w[K,N] * x[M,K]` | 左.i == 右.j | 左是权重、右是输入（权重在左） |
//!
//! 归一之后只认角色，所以两种写法生成同一段代码：`weight.apply_into(&input, &mut y)`
//! （DYN 档为 `apply_dyn_into`）。**同一维必须写同一个下标**（见诊断第 1 条），所以
//! `M`/`K`/`N` 这些名字在三处出现时是一致的。
//!
//! 仍**不做**：`alpha`/`beta` 融合、`+` 多操作数、不带下标的简写里判静态/DYN 两档
//! （`matmul!(x * w)` 按全静态生成）。
//!
//! # 诊断
//!
//! | # | 情况 | 谁报 | 锚点 |
//! |---|---|---|---|
//! | 1 | 同一维两侧下标名不同（`y[M,N] = x[m,K] * …`、`x[M,K] * w[Q,N]`） | 宏 | 两个下标（**双锚定**） |
//! | 2 | 同一次操作数两个下标同名（`y[M, M]`） | 宏 | 两个下标 |
//! | 3 | 下标个数 ≠ 2 / 没写在方括号里 | 宏 | 该操作数 |
//! | 4 | 不是 `a[..] * b[..]` 这个形状（多了 `+`、少了 `*`…） | 宏 | 出错的那个 token |
//! | 5 | 大写下标其实**不是** `const` | rustc | `cannot find value 'M' in this scope` |
//!
//! 第 5 条必须说清楚：**proc-macro 没有名字解析**，它无法知道 `M` 是不是 `const`，所以
//! "大写下标必须是 const"这句提示只能写在**本文档**里，进不了编译错误。用类型标注去逼也
//! 徒劳——名字解析失败发生在任何类型检查之前。
//!
//! # Span 从第一行就保留
//!
//! 生成代码时**复用用户的 token**（`x`、`M`、`K`…），而不是拼一段源码字符串再 parse：
//! 后者的 Span 会全部落在 `call_site()`，形状报错就指不到公式里的具体位置。所以下面
//! `expand` 全程用 `TokenTree` 拼装，只有样板（`let`、`.`、`apply_into`…）用 `call_site()`。
//! 括号必须用 `Group::new(Delimiter::Parenthesis, …)` 表示——`"(".parse::<TokenStream>()`
//! 是**失败**的（不配对的定界符），这一点踩过：早先版本用它拼片段，括号被静默丢掉，
//! 生成的代码语法都不成立。

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

/// 一个下标的档次：编译期维度还是运行期值。**按首字符判定**（见 crate 文档）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// 大写开头：编译期维度（`const`，进类型标注）
    Const,
    /// 小写或下划线开头：运行期值（`usize`，宏自己绑定 + 断言）
    Runtime,
}

impl Kind {
    fn of(name: &str) -> Result<Kind, String> {
        match name.chars().next() {
            Some(c) if c.is_ascii_uppercase() => Ok(Kind::Const),
            Some(c) if c.is_ascii_lowercase() || c == '_' => Ok(Kind::Runtime),
            _ => Err(format!(
                "下标 `{name}` 必须以字母或下划线开头（大写=编译期 const，小写=运行期值）"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Const => "编译期 const",
            Kind::Runtime => "运行期值",
        }
    }
}

/// 一个带两个下标的操作数：`x[M, K]`。
struct Operand {
    name: Ident,
    i: Ident,
    j: Ident,
}

impl Operand {
    fn shape(&self) -> (String, String) {
        (self.i.to_string(), self.j.to_string())
    }
}

/// 解析出来的公式。固定读法：`out[am, bn] = a[am, ak] * b[ak, bn]`。
struct Formula {
    out: Option<Operand>,
    a: Operand,
    b: Operand,
}

/// 一条诊断：主锚点 + 可选第二锚点（双锚定）+ 可选 note。
struct Diagnostic {
    msg: String,
    span: Span,
    second: Option<Span>,
    note: Option<String>,
}

impl Diagnostic {
    fn new(msg: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            msg: msg.into(),
            span,
            second: None,
            note: None,
        }
    }

    fn with_second(mut self, span: Span) -> Self {
        self.second = Some(span);
        self
    }

    fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// 发出**两条** `compile_error!`：位置分别锚在主/次 token 上。
    ///
    /// 为什么不用一条：`compile_error!` 只有一个 Span，而"公式里两处不一致"这类错误，
    /// 用户的直觉是"我哪一侧写错了"——两侧都标出来才不用来回找。
    ///
    /// 整段包在 `{ … }` 里：`matmul!` 会出现在**语句位置**（`matmul!(y[..] = ..);`），
    /// 也会出现在**表达式位置**（`let y = matmul!(..)`）。裸着两条 `compile_error!`
    /// 在表达式位置解析不了（实测报 "expected one of `.`, `;` … found compile_error"），
    /// 包成 block 两种位置都成立。
    fn emit(self) -> TokenStream {
        let mut body = TokenStream::new();
        body.extend(compile_error(&self.msg, self.span));
        body.extend([p(';')]);
        if let Some(second) = self.second {
            let hint = self
                .note
                .clone()
                .unwrap_or_else(|| "（公式里这两处必须一致）".to_string());
            body.extend(compile_error(&hint, second));
            body.extend([p(';')]);
        } else if let Some(note) = &self.note {
            body.extend(compile_error(note, self.span));
            body.extend([p(';')]);
        }
        let mut ts = TokenStream::new();
        ts.extend([group(Delimiter::Brace, body)]);
        ts
    }
}

/* ------------------------------ token 拼装 ------------------------------ */

fn id(s: &str) -> TokenTree {
    TokenTree::Ident(Ident::new(s, Span::call_site()))
}

fn p(c: char) -> TokenTree {
    TokenTree::Punct(Punct::new(c, Spacing::Alone))
}

fn group(delim: Delimiter, inner: TokenStream) -> TokenTree {
    TokenTree::Group(Group::new(delim, inner))
}

/// `compile_error!("…")`，**所有 token 的 Span 都设成 `span`**。
///
/// 只给字符串字面量设 Span 是不够的：rustc 把 `compile_error!` 的诊断定位在它自己那次
/// 宏调用的位置上，也就是 `compile_error` 这个 ident / 整个调用组的位置——那样报错会指向
/// 整条 `matmul!(…)`，而不是出错的那个下标（实测踩过）。`quote_spanned!` 也是这么干的：
/// 把 span 铺满整棵小树。
fn compile_error(msg: &str, span: Span) -> TokenStream {
    let ident = Ident::new("compile_error", span);
    let mut bang = Punct::new('!', Spacing::Alone);
    bang.set_span(span);
    let mut lit = Literal::string(msg);
    lit.set_span(span);
    let mut inner = TokenStream::new();
    inner.extend([TokenTree::Literal(lit)]);
    let mut group = Group::new(Delimiter::Parenthesis, inner);
    group.set_span(span);
    let mut ts = TokenStream::new();
    ts.extend([
        TokenTree::Ident(ident),
        TokenTree::Punct(bang),
        TokenTree::Group(group),
    ]);
    ts
}

/// 样板 token 串：`a.b(&c, &mut d)` 这类调用（接收者与实参里的用户 token 单独传入）。
fn method(recv: &Ident, name: &str, args: TokenStream) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([
        TokenTree::Ident(recv.clone()),
        p('.'),
        id(name),
        group(Delimiter::Parenthesis, args),
    ]);
    ts
}

fn amp(ident: &Ident) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([p('&'), TokenTree::Ident(ident.clone())]);
    ts
}

fn amp_mut(ident: &Ident) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([p('&'), id("mut"), TokenTree::Ident(ident.clone())]);
    ts
}

/// `'_`（借用型视图的标注里要用）。
fn lifetime_underscore() -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([
        TokenTree::Punct(Punct::new('\'', Spacing::Joint)),
        TokenTree::Ident(Ident::new("_", Span::call_site())),
    ]);
    ts
}

/// `let _: &Ty<'_, _, {K}, {N}> = &x;`（元素类型用 `_` 推；用户 token 的 Span 保留）。
///
/// `view` 为真表示该类型带生命周期参数（`Mat`/`MatDyn`），此时泛型表是 `'_, _, 常量…`；
/// 否则是 `_, 常量…`（`MatBuf`/`MatBufDyn`）。顺序错了就会得到 E0747。
fn annotate(ty: &str, view: bool, consts: Vec<TokenTree>, value: &Ident) -> TokenStream {
    let mut generics = TokenStream::new();
    if view {
        generics.extend(lifetime_underscore());
        generics.extend([p(',')]);
    }
    generics.extend([id("_")]);
    for c in consts {
        generics.extend([p(',')]);
        generics.extend([c]);
    }
    let mut ty_ts = TokenStream::new();
    ty_ts.extend([id(ty), p('<')]);
    ty_ts.extend(generics);
    ty_ts.extend([p('>')]);
    let mut ts = TokenStream::new();
    ts.extend([
        id("let"),
        id("_"),
        p(':'),
        p('&'),
        group(Delimiter::None, ty_ts),
        p('='),
        p('&'),
        TokenTree::Ident(value.clone()),
        p(';'),
    ]);
    ts
}

/// `{ K }`：把用户的大写下标当 const 泛型实参（Span 保留，报错落在公式上）。
fn const_arg(ident: &Ident) -> TokenTree {
    let mut inner = TokenStream::new();
    inner.extend([TokenTree::Ident(ident.clone())]);
    group(Delimiter::Brace, inner)
}

/* ------------------------------ 解析 ------------------------------ */

fn parse_operand(tokens: &[TokenTree], at: usize) -> Result<(Operand, usize), Diagnostic> {
    let Some(found) = tokens.get(at) else {
        return Err(Diagnostic::new(
            "公式在这里就结束了，缺少操作数",
            Span::call_site(),
        ));
    };
    let TokenTree::Ident(name) = found else {
        return Err(Diagnostic::new(
            "这里应该是操作数（形如 `x[M, K]`）",
            found.span(),
        ));
    };
    let name = name.clone();
    let name_span = name.span();

    let Some(TokenTree::Group(arg)) = tokens.get(at + 1) else {
        // 紧跟着 `*` 说明这是标量前缀（`alpha * x[..]`）：单独给消息，否则用户只看到
        // "缺少下标"，看不出真正的问题是 v1 还没做 `alpha`/融合。
        if let Some(TokenTree::Punct(star)) = tokens.get(at + 1) {
            if star.as_char() == '*' {
                return Err(Diagnostic::new(
                    format!("v1 不支持标量前缀：`{name} * …`（`alpha` 缩放/融合还没实现）"),
                    name_span,
                )
                .with_note(
                    "要缩放请先算 `matmul!(…)` 再对结果乘标量；`alpha`/`beta` 的语义记在契约里、尚未启用",
                ));
            }
        }
        return Err(Diagnostic::new(
            format!("`{name}` 后面缺少下标：请写成 `{name}[行下标, 列下标]`"),
            name_span,
        ));
    };
    if arg.delimiter() != Delimiter::Bracket {
        return Err(Diagnostic::new(
            format!("`{name}` 的下标必须写在方括号里：`{name}[行下标, 列下标]`"),
            arg.span(),
        ));
    }

    let parts: Vec<TokenTree> = arg.stream().into_iter().collect();
    let (i, j) = match parts.as_slice() {
        [TokenTree::Ident(i), TokenTree::Punct(c), TokenTree::Ident(j)] if c.as_char() == ',' => {
            (i.clone(), j.clone())
        }
        _ => {
            return Err(Diagnostic::new(
                format!("`{name}` 的下标必须是两个标识符：`{name}[行下标, 列下标]`"),
                arg.span(),
            ))
        }
    };
    if i.to_string() == j.to_string() {
        return Err(Diagnostic::new(
            format!("同一次操作数的两个下标不能同名（`{name}[{i}, {i}]`）"),
            i.span(),
        )
        .with_second(j.span())
        .with_note("两个下标必须分别表示行与列"));
    }
    Ok((Operand { name, i, j }, at + 2))
}

fn parse(input: TokenStream) -> Result<Formula, Diagnostic> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let (first, mut at) = parse_operand(&tokens, 0)?;

    let parsed = match tokens.get(at) {
        // `out[..] = a[..] * b[..]`
        Some(TokenTree::Punct(eq)) if eq.as_char() == '=' => {
            at += 1;
            let (lhs, next) = parse_operand(&tokens, at)?;
            at = next;
            match tokens.get(at) {
                Some(TokenTree::Punct(star)) if star.as_char() == '*' => at += 1,
                other => {
                    return Err(Diagnostic::new(
                        "输出与乘积之间缺少 `*`（v1 只接受单项乘积 `y[..] = a[..] * b[..]`）",
                        other.map_or(first.i.span(), TokenTree::span),
                    ))
                }
            }
            let (b, next) = parse_operand(&tokens, at)?;
            at = next;
            (Some(first), lhs, b)
        }
        // `a[..] * b[..]`
        Some(TokenTree::Punct(star)) if star.as_char() == '*' => {
            at += 1;
            let (b, next) = parse_operand(&tokens, at)?;
            at = next;
            (None, first, b)
        }
        other => {
            return Err(Diagnostic::new(
                "公式应该是 `y[行, 列] = a[行, 收缩] * b[收缩, 列]`（或省掉输出）",
                other.map_or(first.i.span(), TokenTree::span),
            ))
        }
    };

    if let Some(trailing) = tokens.get(at) {
        return Err(Diagnostic::new(
            "公式到乘积就结束了（v1 不支持 `+`/`alpha`/多操作数，见 crate 文档）",
            trailing.span(),
        ));
    }
    let (out, left, right) = parsed;
    normalize(out, left, right)
}

/// 角色归一（**纯名字层**：能单元测试）。
///
/// 两个操作数有两种合法读法，靠"共享下标出现在哪一侧"区分：
///
/// - `x[M,K] * w[K,N]`：左.j == 右.i → 左是**输入**、右是**权重**（= `A·B`）
/// - `w[K,N] * x[M,K]`：左.i == 右.j → 左是**权重**、右是**输入**（权重在左）
///
/// 返回 `true` 表示需要交换（权重在左）。
fn roles(left: (&str, &str), right: (&str, &str)) -> Result<bool, ()> {
    if left.1 == right.0 {
        Ok(false)
    } else if left.0 == right.1 {
        Ok(true)
    } else {
        Err(())
    }
}

/// 按 [`roles`] 的判定把 `a`/`b` 归一成 **input / weight**，下游的检查与生成都只认角色。
fn normalize(out: Option<Operand>, left: Operand, right: Operand) -> Result<Formula, Diagnostic> {
    let (li, lj) = left.shape();
    let (ri, rj) = right.shape();
    match roles((li.as_str(), lj.as_str()), (ri.as_str(), rj.as_str())) {
        Ok(false) => Ok(Formula {
            out,
            a: left,
            b: right,
        }),
        Ok(true) => Ok(Formula {
            out,
            a: right,
            b: left,
        }),
        Err(()) => Err(Diagnostic::new(
            format!("收缩维的下标不一致：左侧操作数写 `{lj}`，右侧操作数写 `{ri}`"),
            left.j.span(),
        )
        .with_second(right.i.span())
        .with_note("两种读法都对不上：`x[M,K] * w[K,N]`（A·B）或 `w[K,N] * x[M,K]`（权重在左）")),
    }
}

/* ------------------------------ 语义检查 ------------------------------ */

/// 语义违规（**纯名字层**：不碰 `proc_macro` 类型，所以能单元测试）。
#[derive(Debug, PartialEq)]
enum Violation {
    /// 同一维两侧写的下标名不同。
    SameDimDifferentNames {
        dim: &'static str,
        lhs: String,
        rhs: String,
        lhs_op: &'static str,
        rhs_op: &'static str,
    },
}

/// 纯逻辑层：**同一维必须写同一个下标**（行、收缩、列各查一次）。
fn check_names(
    out: Option<(&str, &str)>,
    a: (&str, &str),
    b: (&str, &str),
) -> Result<(), Violation> {
    let mut pairs: Vec<(&'static str, &str, &str, &'static str, &'static str)> =
        vec![("收缩", a.1, b.0, "左侧操作数", "右侧操作数")];
    if let Some((out_i, out_j)) = out {
        pairs.push(("行", out_i, a.0, "输出", "左侧操作数"));
        pairs.push(("列", out_j, b.1, "输出", "右侧操作数"));
    }
    for (dim, lhs, rhs, lhs_op, rhs_op) in pairs {
        if lhs != rhs {
            return Err(Violation::SameDimDifferentNames {
                dim,
                lhs: lhs.to_string(),
                rhs: rhs.to_string(),
                lhs_op,
                rhs_op,
            });
        }
    }
    Ok(())
}

/// 在操作数里按名字找下标 token（用于把违规锚到具体位置）。
fn span_of(op: &Operand, name: &str) -> Span {
    if op.i.to_string() == name {
        op.i.span()
    } else {
        op.j.span()
    }
}

fn check(f: &Formula) -> Result<(), Diagnostic> {
    let out_names = f.out.as_ref().map(Operand::shape);
    let a_names = f.a.shape();
    let b_names = f.b.shape();
    match check_names(
        out_names.as_ref().map(|(i, j)| (i.as_str(), j.as_str())),
        (a_names.0.as_str(), a_names.1.as_str()),
        (b_names.0.as_str(), b_names.1.as_str()),
    ) {
        Ok(()) => Ok(()),
        Err(Violation::SameDimDifferentNames {
            dim,
            lhs,
            rhs,
            lhs_op,
            rhs_op,
        }) => {
            let lhs_operand = match lhs_op {
                "输出" => f.out.as_ref().expect("输出维检查的前提是有输出"),
                _ => &f.a,
            };
            let rhs_operand = if rhs_op == "右侧操作数" {
                &f.b
            } else {
                &f.a
            };
            let kinds = match (Kind::of(&lhs), Kind::of(&rhs)) {
                (Ok(l), Ok(r)) if l != r => {
                    format!("`{lhs}` 是{}、`{rhs}` 是{}：档次也不同", l.name(), r.name())
                }
                _ => format!("两处必须写同一个下标（当前是 `{lhs}` 与 `{rhs}`）"),
            };
            Err(Diagnostic::new(
                format!("{dim}维的下标不一致：{lhs_op}写 `{lhs}`，{rhs_op}写 `{rhs}`"),
                span_of(lhs_operand, &lhs),
            )
            .with_second(span_of(rhs_operand, &rhs))
            .with_note(format!(
                "{kinds}。大写=编译期 const、小写=运行期值；同一维必须一致"
            )))
        }
    }
}

/* ------------------------------ 生成 ------------------------------ */

/// 生成代码。**复用用户的 token**（Span 保留），只有样板用 `call_site()`。
fn expand(f: &Formula) -> TokenStream {
    let runtime_rows = matches!(Kind::of(&f.a.i.to_string()), Ok(Kind::Runtime));

    match &f.out {
        // 写进已有输出
        Some(out) => {
            let mut body = TokenStream::new();
            // 前三行是形状标注（大写下标）或运行期绑定 + 断言（小写下标）
            if runtime_rows {
                body.extend([id("let"), TokenTree::Ident(f.a.i.clone()), p('=')]);
                body.extend(method(&f.a.name, "rows", TokenStream::new()));
                body.extend([p(';')]);
                let mut args = TokenStream::new();
                args.extend([
                    TokenTree::Ident(out.name.clone()), // ← 操作数名（yd），不是下标名（m）
                    p('.'),
                    id("rows"),
                    group(Delimiter::Parenthesis, TokenStream::new()),
                    p(','),
                    TokenTree::Ident(f.a.i.clone()),
                ]);
                body.extend([
                    id("assert_eq"),
                    p('!'),
                    group(Delimiter::Parenthesis, args),
                    p(';'),
                ]);
                body.extend(annotate("MatDyn", true, vec![const_arg(&f.a.j)], &f.a.name));
                body.extend(annotate(
                    "Mat",
                    true,
                    vec![const_arg(&f.b.i), const_arg(&f.b.j)],
                    &f.b.name,
                ));
                body.extend(annotate(
                    "MatBufDyn",
                    false,
                    vec![const_arg(&f.b.j)],
                    &out.name,
                ));
                let mut args = TokenStream::new();
                args.extend(amp(&f.a.name));
                args.extend([p(',')]);
                args.extend(amp_mut(&out.name));
                body.extend(method(&f.b.name, "apply_dyn_into", args));
            } else {
                body.extend(annotate(
                    "Mat",
                    true,
                    vec![const_arg(&f.a.i), const_arg(&f.a.j)],
                    &f.a.name,
                ));
                body.extend(annotate(
                    "Mat",
                    true,
                    vec![const_arg(&f.b.i), const_arg(&f.b.j)],
                    &f.b.name,
                ));
                body.extend(annotate(
                    "MatBuf",
                    false,
                    vec![const_arg(&out.i), const_arg(&out.j)],
                    &out.name,
                ));
                let mut args = TokenStream::new();
                args.extend(amp(&f.b.name));
                args.extend([p(',')]);
                args.extend(amp_mut(&out.name));
                // 静态档写进已有输出：`Mat::mul_into`（`apply_into` 在 `Prepared` 上）
                body.extend(method(&f.a.name, "mul_into", args));
            }
            // 包进 block：不污染调用点作用域，也让整条公式是一个表达式
            let mut ts = TokenStream::new();
            ts.extend([group(Delimiter::Brace, body)]);
            ts
        }
        // 只写乘积：输出新分配（无下标时无法判档，按全静态生成；DYN 请写下标形式）
        None => {
            if runtime_rows {
                let mut args = TokenStream::new();
                args.extend([amp(&f.a.name)]);
                method(&f.b.name, "apply_dyn", args)
            } else {
                let mut args = TokenStream::new();
                args.extend([amp(&f.b.name)]);
                method(&f.a.name, "mul", args)
            }
        }
    }
}

/// 公式 DSL：`matmul!(y[M, N] = x[M, K] * w[K, N])`。
///
/// 生成什么、接受什么、报错怎么锚，见 crate 文档（`cargo doc -p lasx_rs_macros`）。
#[proc_macro]
pub fn matmul(input: TokenStream) -> TokenStream {
    match parse(input).and_then(|f| check(&f).map(|()| f)) {
        Ok(f) => expand(&f),
        Err(d) => d.emit(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 首字符分派是这套 DSL 的核心规则，单独钉住（含 `_m`、`m2`、`M_` 三个边界）。
    #[test]
    fn kind_is_decided_by_the_first_char() {
        for name in ["M", "N", "K", "M_", "K1"] {
            assert_eq!(Kind::of(name), Ok(Kind::Const), "{name}");
        }
        for name in ["m", "n", "m2", "_m", "_"] {
            assert_eq!(Kind::of(name), Ok(Kind::Runtime), "{name}");
        }
        assert!(Kind::of("1m").is_err(), "首字符不是字母/下划线要报错");
    }

    /// 两种读法的角色判定：`A·B` 与"权重在左"都要认，且对不上要报错。
    #[test]
    fn roles_accept_both_readings() {
        // A·B：左.j == 右.i
        assert_eq!(roles(("M", "K"), ("K", "N")), Ok(false));
        // 权重在左：左.i == 右.j
        assert_eq!(roles(("K", "N"), ("M", "K")), Ok(true));
        // 都对不上
        assert_eq!(roles(("M", "K"), ("Q", "N")), Err(()));
        assert_eq!(roles(("M", "K"), ("N", "K")), Err(()));
    }

    /// 同一维必须同名：三处（行、收缩、列）各一个反例 + 两个正例。
    #[test]
    fn same_dimension_must_use_the_same_name() {
        // 收缩维
        let v = check_names(Some(("M", "N")), ("M", "K"), ("Q", "N")).unwrap_err();
        assert!(matches!(
            v,
            Violation::SameDimDifferentNames { dim: "收缩", .. }
        ));
        // 行维（输出大写、输入小写：档次也不同，提示里会点出来）
        let v = check_names(Some(("M", "N")), ("m", "K"), ("K", "N")).unwrap_err();
        let Violation::SameDimDifferentNames { dim, lhs, rhs, .. } = v;
        assert_eq!(dim, "行");
        assert_eq!((lhs.as_str(), rhs.as_str()), ("M", "m"));
        // 列维
        let v = check_names(Some(("M", "N")), ("M", "K"), ("K", "n")).unwrap_err();
        assert!(matches!(
            v,
            Violation::SameDimDifferentNames { dim: "列", .. }
        ));
        // 正例：全静态、全运行期都合法
        assert!(check_names(Some(("M", "N")), ("M", "K"), ("K", "N")).is_ok());
        assert!(check_names(Some(("m", "N")), ("m", "K"), ("K", "N")).is_ok());
        // 无输出时只查收缩维
        assert!(check_names(None, ("M", "K"), ("K", "N")).is_ok());
        assert!(check_names(None, ("M", "K"), ("Q", "N")).is_err());
    }
}
