//! `lasx_rs` 的公式 DSL：把矩阵乘写成数学式子，形状编译期对齐。
//!
//! ```text
//! matmul!(y[M, N] = x[M, K] * w[K, N]);     // 全静态：M/K/N 是 const
//! matmul!(y[m, N] = x[m, K] * w[K, N]);     // DYN：m 是运行期行数，K/N 仍是 const
//! matmul!(y[M, N] = x[M, K] * p);           // 右操作数是 Prepared plan（不带方括号）
//! matmul!(y[M, N] = alpha * x[M, K] * w[K, N]);  // 标量前缀：C = alpha·A·B（beta = 0）
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
//! 这是选 proc-macro 而不是 `macro_rules!` 的第一条理由：`macro_rules!` 拿不到 token 的
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
//!         | ident                                   ← 只在乘积右侧：视为 Prepared plan
//! 标量   := ident | 字面量                          ← 可选前缀，写在乘积最前面
//! ```
//!
//! 两个**带下标**的操作数按"共享下标出现在哪一侧"判角色，两种读法都认：
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
//! 仍**不做**：`beta`（累加/原地）的 DSL 写法——那需要输出是输入之一，与"输入输出不能是
//! 同一块内存"的契约冲突，请直接用 `api::matmul_scaled_into`（`C = alpha·A·B + beta·C`）；
//! `+` 多操作数、不带下标的简写里判静态/DYN 两档（`matmul!(x * w)` 按全静态生成）。
//!
//! # 标量前缀（`alpha`）
//!
//! 乘积最前面可以写一个标量（`matmul!(y[M,N] = alpha * x[M,K] * w[K,N])`），语义是
//! **`y = alpha·(A·B)`**，即 CBLAS 的 `alpha` 配 `beta = 0`（输出被完全覆盖、旧值不读）：
//!
//! ```text
//! matmul!(y[M, N] = alpha * x[M, K] * w[K, N]);  →  x.mul_scaled_into(alpha, &w, &mut y)
//! matmul!(y[M, N] = alpha * x[M, K] * p);        →  p.apply_scaled_into(alpha, &x, &mut y)
//! matmul!(y[m, N] = alpha * x[m, K] * w[K, N]);  →  w.apply_scaled_dyn_into(alpha, &x, &mut y)
//! matmul!(alpha * x[M, K] * w[K, N]);            →  x.mul_scaled(alpha, &w)
//! ```
//!
//! 标量 token **原样进生成的代码**，所以可以是局部变量、`const` 或字面量（求值为 `f32`）。
//!
//! 两条边界是刻意的：
//!
//! - **只对 `f32` 成立**。融合内核（`ops::matmul::Epi::Scale`）只在 f32 路径上；`f64` 写标量
//!   前缀得到的是"方法不存在"的编译错误，而不是静默退化成两遍法——`f64` 要缩放请先
//!   `mul` 再整体乘。
//! - **`beta` 没有 DSL 写法**（见上）。
//!
//! 数值上 `alpha` 只是从"算完再整体乘"搬进输出那一遍，累加次序没动 ⇒ 与两遍法**逐位一致**
//! （`shape.rs` 的 `test_formula_dsl_scaled_matches_two_pass_bit_for_bit` 钉住）。
//!
//! # 直接吃 `Prepared`（plan 形态）
//!
//! `Mat::mul_into` 的实现是 `rhs.prepare().apply_into(self, out)`：**每次调用都重新打包
//! 右操作数**。层循环里复用同一权重时这是纯浪费（256³ 实测 143.7 µs → 96.5–99.1 µs，
//! 省 **31–34%**）。所以给 DSL 一个"**右操作数不写方括号**"的形态——`Prepared` 自带形状，
//! 方括号的有无就是区分信号：
//!
//! ```text
//! matmul!(y[M, N] = x[M, K] * p);   →  p.apply_into(&x, &mut y)         // const 行数
//! matmul!(y[m, N] = x[m, K] * p);   →  p.apply_dyn_into(&x, &mut y)     // 运行期行数
//! matmul!(x[M, K] * p)              →  p.apply(&x) / p.apply_dyn(&x)    // 无输出、新分配
//! ```
//!
//! `p` 由调用方**在循环外**建好，打包只发生一次：
//!
//! ```text
//! let p = w.prepare();                        // 循环外一次
//! for _ in 0..layers { matmul!(y[M, N] = x[M, K] * p); }
//! ```
//!
//! 三条边界，都是刻意的：
//!
//! - **plan 只能写在乘积右侧**。`p * x[M,K]` 与"带方括号的输入写在右侧"在 token 层无法与
//!   标量前缀区分，于是留一条"形状必须写在左侧"的硬规则（plan 形态只认右操作数）。
//! - **权重形状不进公式**。K/N 由 `Prepared` 自己的 const 参数定，与 `x` / `y` 的类型标注
//!   对接（写错仍是编译错误：`x` 的列、`y` 的列分别与 plan 的 K/N 对齐）。
//! - **不做隐式/指纹缓存**。`.prepare()` 返回的 `Prepared` 借用了权重切片，只有**显式**
//!   plan 才是可证明安全的；按指针/指纹键做透明缓存会在缓冲区复用时给出**陈旧结果**
//!   （静默错结果），与位精确承诺冲突。
//!
//! **位精确**：这条形态与手写 `let p = w.prepare(); p.apply_into(&x, &mut y)` 生成的是
//! **同一段调用**（宏不额外做任何事），所以逐位一致是构造性的，不是巧合。
//!
//! # 诊断
//!
//! | # | 情况 | 谁报 | 锚点 |
//! |---|---|---|---|
//! | 1 | 同一维两侧下标名不同（`y[M,N] = x[m,K] * …`、`x[M,K] * w[Q,N]`） | 宏 | 两个下标（**双锚定**） |
//! | 2 | 同一次操作数两个下标同名（`y[M, M]`） | 宏 | 两个下标 |
//! | 3 | 操作数没写方括号（`y[M,N] = x * w[K,N]`）、或 plan 写在乘积左侧 | 宏 | 该操作数 |
//! | 4 | 下标个数 ≠ 2 / 下标没写在方括号里（`x[]`、`x(M, K)`） | 宏 | 该操作数 |
//! | 5 | 不是 `a[..] * b[..]` 这个形状（多了 `+`、少了 `*`…） | 宏 | 出错的那个 token |
//! | 6 | 大写下标其实**不是** `const` | rustc | `cannot find value 'M' in this scope` |
//!
//! 第 6 条必须说清楚：**proc-macro 没有名字解析**，它无法知道 `M` 是不是 `const`，所以
//! "大写下标必须是 const"这句提示只能写在**本文档**里，进不了编译错误。用类型标注去逼也
//! 徒劳——名字解析失败发生在任何类型检查之前。
//!
//! 第 3 条只对"没写方括号、也不在 plan 位置"的操作数成立：形如 `ident * …` 的**最前面**
//! 那个 ident 已经被标量前缀的解析（源码里的 `parse_scalar`）吃掉了，不会走到这条诊断。
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

mod vec;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

/// 一个下标的档次：编译期维度还是运行期值。**按首字符判定**（见 crate 文档）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
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

/// 一个操作数：`x[M, K]`（带方括号，形状写进公式）或 `p`（不带方括号 ⇒ `Prepared` plan）。
///
/// 下标是 `Option`：`None` 就是 plan——`Prepared` 自带形状，所以不需要（也不能）在公式里写。
/// 解析保证"输入 / 输出"这两个角色一定有下标（见 [`parse_operand`] 的 `allow_plan`），
/// 所以 [`Operand::i`] / [`Operand::j`] 对它们是安全的。
struct Operand {
    name: Ident,
    dims: Option<(Ident, Ident)>,
}

impl Operand {
    /// 不带方括号的操作数：视为 `Prepared` plan。
    fn plan(name: Ident) -> Operand {
        Operand { name, dims: None }
    }

    fn is_plan(&self) -> bool {
        self.dims.is_none()
    }

    /// 形状（名字层）。`None` = plan（形状在运行期由 plan 自带）。
    fn shape(&self) -> Option<(String, String)> {
        self.dims
            .as_ref()
            .map(|(i, j)| (i.to_string(), j.to_string()))
    }

    fn dims(&self) -> &(Ident, Ident) {
        self.dims
            .as_ref()
            .expect("该角色由解析保证带下标（输入/输出）")
    }

    /// 行下标（编译期 const 或运行期值由首字符决定）。
    fn i(&self) -> &Ident {
        &self.dims().0
    }

    /// 列下标。
    fn j(&self) -> &Ident {
        &self.dims().1
    }
}

/// 解析出来的公式，已按角色归一：`input` 是输入、`weight` 是权重（可能是 plan）。
///
/// `weight.is_plan()` 为真表示宏吃到的是 [`Prepared`]：生成 `weight.apply_into(&input, …)`
/// 而不是 `input.mul_into(&weight, …)`——后者内部每次都会 `prepare()`。
///
/// `scalar` 是标量前缀的 token（`matmul!(y = alpha * x * w)` 里的 `alpha`）。它原样进生成的
/// 代码，所以可以是任何求值为 `f32` 的表达式（局部变量、`const`、字面量）；语义是
/// `out = alpha·(A·B)`，即 `beta = 0`（`out` 被完全覆盖、旧值不读）。
struct Formula {
    out: Option<Operand>,
    input: Operand,
    weight: Operand,
    scalar: Option<TokenTree>,
}

/// 一条诊断：主锚点 + 可选第二锚点（双锚定）+ 可选 note。
pub(crate) struct Diagnostic {
    msg: String,
    span: Span,
    second: Option<Span>,
    note: Option<String>,
}

impl Diagnostic {
    pub(crate) fn new(msg: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            msg: msg.into(),
            span,
            second: None,
            note: None,
        }
    }

    pub(crate) fn with_second(mut self, span: Span) -> Self {
        self.second = Some(span);
        self
    }

    pub(crate) fn with_note(mut self, note: impl Into<String>) -> Self {
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
    pub(crate) fn emit(self) -> TokenStream {
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

pub(crate) fn id(s: &str) -> TokenTree {
    TokenTree::Ident(Ident::new(s, Span::call_site()))
}

pub(crate) fn p(c: char) -> TokenTree {
    TokenTree::Punct(Punct::new(c, Spacing::Alone))
}

pub(crate) fn group(delim: Delimiter, inner: TokenStream) -> TokenTree {
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
pub(crate) fn method(recv: &Ident, name: &str, args: TokenStream) -> TokenStream {
    let mut ts = TokenStream::new();
    ts.extend([
        TokenTree::Ident(recv.clone()),
        p('.'),
        id(name),
        group(Delimiter::Parenthesis, args),
    ]);
    ts
}

pub(crate) fn amp(ident: &Ident) -> TokenStream {
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
pub(crate) fn const_arg(ident: &Ident) -> TokenTree {
    let mut inner = TokenStream::new();
    inner.extend([TokenTree::Ident(ident.clone())]);
    group(Delimiter::Brace, inner)
}

/* ------------------------------ 解析 ------------------------------ */

/// 解析一个操作数。
///
/// `allow_plan` 只对**乘积右侧**的操作数放行"不带方括号"的 plan 形态：`Prepared` 自带形状，
/// 所以那里不写方括号就是"这是打包好的权重"。左侧必须写形状——`alpha * x[M,K]` 与
/// `p * x[M,K]` 在 token 层同形，只有把 plan 限死在右侧，标量前缀那条诊断才留得住。
fn parse_operand(
    tokens: &[TokenTree],
    at: usize,
    allow_plan: bool,
) -> Result<(Operand, usize), Diagnostic> {
    let Some(found) = tokens.get(at) else {
        return Err(Diagnostic::new(
            "公式在这里就结束了，缺少操作数",
            Span::call_site(),
        ));
    };
    let TokenTree::Ident(name) = found else {
        return Err(Diagnostic::new(
            "这里应该是操作数（形如 `x[M, K]`；乘积右侧还可以是不带方括号的 `p`，即 `Prepared`）",
            found.span(),
        ));
    };
    let name = name.clone();
    let name_span = name.span();

    // 后面不是一个方括号组：要么是 plan 形态（右侧），要么是缺下标 / 标量前缀（左侧）。
    let arg = match tokens.get(at + 1) {
        Some(TokenTree::Group(arg)) => arg,
        other => {
            if allow_plan {
                return Ok((Operand::plan(name), at + 1));
            }
            // 紧跟着 `*` 说明这个操作数少了方括号（合法操作数必须写 `x[行, 列]`；标量前缀
            // 是 `alpha * x[..] * w[..]`，标量在最前面，已经在 `parse_scalar` 里吃掉）。
            if let Some(TokenTree::Punct(star)) = other
                && star.as_char() == '*'
            {
                return Err(Diagnostic::new(
                    format!("`{name}` 后面缺少下标：请写成 `{name}[行下标, 列下标]`"),
                    name_span,
                )
                .with_note(
                    "标量前缀要写成 `alpha * x[M, K] * w[K, N]`（标量在最前面）——\
                     `alpha` 会走融合入口，等价于 `C = alpha·A·B`（`beta = 0`）",
                ));
            }
            return Err(Diagnostic::new(
                format!("`{name}` 后面缺少下标：请写成 `{name}[行下标, 列下标]`"),
                name_span,
            ));
        }
    };
    if arg.delimiter() != Delimiter::Bracket {
        return Err(Diagnostic::new(
            format!("`{name}` 的下标必须写在方括号里：`{name}[行下标, 列下标]`"),
            arg.span(),
        ));
    }

    let parts: Vec<TokenTree> = arg.stream().into_iter().collect();
    let (i, j) = match parts.as_slice() {
        [
            TokenTree::Ident(i),
            TokenTree::Punct(c),
            TokenTree::Ident(j),
        ] if c.as_char() == ',' => (i.clone(), j.clone()),
        _ => {
            return Err(Diagnostic::new(
                format!("`{name}` 的下标必须是两个标识符：`{name}[行下标, 列下标]`"),
                arg.span(),
            ));
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
    Ok((
        Operand {
            name,
            dims: Some((i, j)),
        },
        at + 2,
    ))
}

fn parse(input: TokenStream) -> Result<Formula, Diagnostic> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    // 标量前缀可以出现在最前面（无输出形态：`alpha * x[M,K] * w[K,N]`）或 `=` 之后。
    // 识别条件是"标识符/字面量后面直接跟 `*`"：合法操作数一定写方括号，所以这个形状只可能
    // 是标量（`Prepared` 形态只允许出现在乘积右侧，不受影响）。
    let (scalar0, mut at) = parse_scalar(&tokens, 0)?;
    // 左侧（输出、输入）必须写形状；只有乘积右侧允许 plan 形态。
    let (first, next) = parse_operand(&tokens, at, false)?;
    at = next;

    let (scalar, parsed) = match tokens.get(at) {
        // `out[..] = a[..] * b[..]`（`=` 之后可以再有标量前缀）
        Some(TokenTree::Punct(eq)) if eq.as_char() == '=' => {
            at += 1;
            let (scalar1, next) = parse_scalar(&tokens, at)?;
            at = next;
            if scalar0.is_some() && scalar1.is_some() {
                return Err(Diagnostic::new(
                    "标量前缀只能写一次",
                    scalar1.as_ref().map_or(first.i().span(), TokenTree::span),
                )
                .with_note("把标量写在 `=` 之后：`y[M, N] = alpha * x[M, K] * w[K, N]`"));
            }
            let (lhs, next) = parse_operand(&tokens, at, false)?;
            at = next;
            match tokens.get(at) {
                Some(TokenTree::Punct(star)) if star.as_char() == '*' => at += 1,
                other => {
                    // `=` 之后已经吃到一个标量前缀、却只解析出一项：`y[M,N] = x * w[K,N]`
                    // 就是这条（`x` 被当成了标量）。这时把三种可能一次说清，否则用户只会
                    // 看到"缺少 `*`"——那正是他写对的地方。
                    if let Some(scalar) = &scalar1 {
                        return Err(Diagnostic::new(
                            format!(
                                "`=` 之后只解析出一项乘积：`{scalar}` 若是输入矩阵，请补下标\
                                 （`x[行, 列]`）；若是 `Prepared` plan，它只能写在乘积右侧\
                                 （`x[M, K] * p`）；若确实是标量前缀，公式还缺第二项\
                                 （`alpha * x[M, K] * w[K, N]`）"
                            ),
                            scalar.span(),
                        ));
                    }
                    return Err(Diagnostic::new(
                        "输出与乘积之间缺少 `*`（v1 只接受单项乘积 `y[..] = a[..] * b[..]`）",
                        other.map_or(first.i().span(), TokenTree::span),
                    ));
                }
            }
            let (b, next) = parse_operand(&tokens, at, true)?;
            at = next;
            (scalar0.or(scalar1), (Some(first), lhs, b))
        }
        // `a[..] * b[..]`（标量前缀已经在最前面吃掉）
        Some(TokenTree::Punct(star)) if star.as_char() == '*' => {
            at += 1;
            let (b, next) = parse_operand(&tokens, at, true)?;
            at = next;
            (scalar0, (None, first, b))
        }
        other => {
            return Err(Diagnostic::new(
                "公式应该是 `y[行, 列] = a[行, 收缩] * b[收缩, 列]`（或省掉输出）",
                other.map_or(first.i().span(), TokenTree::span),
            ));
        }
    };

    if let Some(trailing) = tokens.get(at) {
        return Err(Diagnostic::new(
            "公式到乘积就结束了（v1 不支持 `+`/多操作数，见 crate 文档）",
            trailing.span(),
        ));
    }
    let (out, left, right) = parsed;
    normalize(out, left, right, scalar)
}

/// 可选的标量前缀 `alpha * …`：返回 `(标量 token, 新位置)`。
///
/// 只有"标识符或字面量 + `*`"才算标量——`x[M, K]` 的 `x` 后面跟的是方括号组，落不进这里。
fn parse_scalar(tokens: &[TokenTree], at: usize) -> Result<(Option<TokenTree>, usize), Diagnostic> {
    let Some(found) = tokens.get(at) else {
        return Ok((None, at));
    };
    if !matches!(found, TokenTree::Ident(_) | TokenTree::Literal(_)) {
        return Ok((None, at));
    }
    match tokens.get(at + 1) {
        Some(TokenTree::Punct(s)) if s.as_char() == '*' => Ok((Some(found.clone()), at + 2)),
        _ => Ok((None, at)),
    }
}

/// 角色归一（**纯名字层**：能单元测试）。
///
/// 两个**带下标**的操作数有两种合法读法，靠"共享下标出现在哪一侧"区分：
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

/// 按角色归一成 **input / weight**，下游的检查与生成都只认角色。
///
/// 右侧不带下标（plan）时角色由**位置**直接定下：右侧是打包好的权重、左侧是输入，
/// 不需要（也无法）用共享下标判——`Prepared` 的形状不在公式里。
fn normalize(
    out: Option<Operand>,
    left: Operand,
    right: Operand,
    scalar: Option<TokenTree>,
) -> Result<Formula, Diagnostic> {
    if right.is_plan() {
        if left.is_plan() {
            return Err(Diagnostic::new(
                "plan 形态只能有一个操作数：右侧是不带方括号的 `p`，左侧必须写形状 `x[M, K]`",
                right.name.span(),
            )
            .with_second(left.name.span())
            .with_note("生成的是 `p.apply_into(&x, &mut y)`，需要一个带形状的输入"));
        }
        return Ok(Formula {
            out,
            input: left,
            weight: right,
            scalar,
        });
    }
    if left.is_plan() {
        // `parse` 只对右侧放行 plan，这里防御性报错（这个函数是纯函数、也能单测）。
        return Err(Diagnostic::new(
            "plan 形态（不带方括号的 `p`）只能写在乘积右侧：`x[M, K] * p`",
            left.name.span(),
        )
        .with_note("左侧写形状后才能用共享下标判角色"));
    }
    let (li, lj) = left.shape().expect("非 plan 必有下标");
    let (ri, rj) = right.shape().expect("非 plan 必有下标");
    match roles((li.as_str(), lj.as_str()), (ri.as_str(), rj.as_str())) {
        Ok(false) => Ok(Formula {
            out,
            input: left,
            weight: right,
            scalar,
        }),
        Ok(true) => Ok(Formula {
            out,
            input: right,
            weight: left,
            scalar,
        }),
        Err(()) => Err(Diagnostic::new(
            format!("收缩维的下标不一致：左侧操作数写 `{lj}`，右侧操作数写 `{ri}`"),
            left.j().span(),
        )
        .with_second(right.i().span())
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
///
/// `weight` 为 `None` 表示 plan 形态：权重形状不在公式里（`Prepared` 自带），所以收缩维与
/// 列维**没有可对照的名字**，只查"输出行 == 输入行"这一条。K/N 是否对得上由类型系统守
/// （plan 的 const 参数与 `x`/`y` 的类型标注对接）。
fn check_names(
    out: Option<(&str, &str)>,
    input: (&str, &str),
    weight: Option<(&str, &str)>,
) -> Result<(), Violation> {
    let mut pairs: Vec<(&'static str, &str, &str, &'static str, &'static str)> = Vec::new();
    if let Some((w_i, _)) = weight {
        pairs.push(("收缩", input.1, w_i, "左侧操作数", "右侧操作数"));
    }
    if let Some((out_i, out_j)) = out {
        pairs.push(("行", out_i, input.0, "输出", "左侧操作数"));
        if let Some((_, w_j)) = weight {
            pairs.push(("列", out_j, w_j, "输出", "右侧操作数"));
        }
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
    if op.i().to_string() == name {
        op.i().span()
    } else {
        op.j().span()
    }
}

fn check(f: &Formula) -> Result<(), Diagnostic> {
    let out_names = f.out.as_ref().and_then(Operand::shape);
    let input_names = f.input.shape().expect("输入必有下标");
    let weight_names = f.weight.shape();
    match check_names(
        out_names.as_ref().map(|(i, j)| (i.as_str(), j.as_str())),
        (input_names.0.as_str(), input_names.1.as_str()),
        weight_names.as_ref().map(|(i, j)| (i.as_str(), j.as_str())),
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
                _ => &f.input,
            };
            let rhs_operand = if rhs_op == "右侧操作数" {
                &f.weight
            } else {
                &f.input
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
///
/// plan 形态（`weight.is_plan()`）与非 plan 形态只差两处：少一条权重类型标注、
/// 接收者换成权重（`weight.apply_into(&input, &mut out)` 而不是 `input.mul_into(&weight, &mut out)`）。
///
/// 有标量前缀（`f.scalar`：`matmul!(y = alpha * x * w)`）时，方法名换成 `*_scaled*` 并把标量
/// token 作为**第一个实参**——底层是 `ops::matmul` 的 `Epi::Scale`，`alpha` 进输出那一遍，
/// 不再单独跑一遍 elementwise（见 `shape.rs` 里那批 `*_scaled*` 方法）。
fn expand(f: &Formula) -> TokenStream {
    let input = &f.input;
    let weight = &f.weight;
    let planned = weight.is_plan();
    let scaled = f.scalar.is_some();
    let runtime_rows = matches!(Kind::of(&input.i().to_string()), Ok(Kind::Runtime));

    match &f.out {
        // 写进已有输出
        Some(out) => {
            let mut body = TokenStream::new();
            // 前三行是形状标注（大写下标）或运行期绑定 + 断言（小写下标）
            if runtime_rows {
                body.extend([id("let"), TokenTree::Ident(input.i().clone()), p('=')]);
                body.extend(method(&input.name, "rows", TokenStream::new()));
                body.extend([p(';')]);
                let mut args = TokenStream::new();
                args.extend([
                    TokenTree::Ident(out.name.clone()), // ← 操作数名（yd），不是下标名（m）
                    p('.'),
                    id("rows"),
                    group(Delimiter::Parenthesis, TokenStream::new()),
                    p(','),
                    TokenTree::Ident(input.i().clone()),
                ]);
                body.extend([
                    id("assert_eq"),
                    p('!'),
                    group(Delimiter::Parenthesis, args),
                    p(';'),
                ]);
                body.extend(annotate(
                    "MatDyn",
                    true,
                    vec![const_arg(input.j())],
                    &input.name,
                ));
                if !planned {
                    body.extend(annotate(
                        "Mat",
                        true,
                        vec![const_arg(weight.i()), const_arg(weight.j())],
                        &weight.name,
                    ));
                }
                // 非 plan：列宽取权重下标（与原实现逐字一致）；plan：权重没下标，取输出下标，
                // 它与 plan 自己的 N 由类型系统对齐。
                let n_col = if planned { out.j() } else { weight.j() };
                body.extend(annotate(
                    "MatBufDyn",
                    false,
                    vec![const_arg(n_col)],
                    &out.name,
                ));
                let mut args = TokenStream::new();
                alpha_arg(&mut args, f.scalar.as_ref());
                args.extend(amp(&input.name));
                args.extend([p(',')]);
                args.extend(amp_mut(&out.name));
                let name = if scaled {
                    "apply_scaled_dyn_into"
                } else {
                    "apply_dyn_into"
                };
                body.extend(method(&weight.name, name, args));
            } else {
                body.extend(annotate(
                    "Mat",
                    true,
                    vec![const_arg(input.i()), const_arg(input.j())],
                    &input.name,
                ));
                if !planned {
                    body.extend(annotate(
                        "Mat",
                        true,
                        vec![const_arg(weight.i()), const_arg(weight.j())],
                        &weight.name,
                    ));
                }
                body.extend(annotate(
                    "MatBuf",
                    false,
                    vec![const_arg(out.i()), const_arg(out.j())],
                    &out.name,
                ));
                let mut args = TokenStream::new();
                alpha_arg(&mut args, f.scalar.as_ref());
                if planned {
                    // plan：打包已完成，直接吃 `Prepared::apply_scaled_into` / `apply_into`
                    args.extend(amp(&input.name));
                    args.extend([p(',')]);
                    args.extend(amp_mut(&out.name));
                    let name = if scaled {
                        "apply_scaled_into"
                    } else {
                        "apply_into"
                    };
                    body.extend(method(&weight.name, name, args));
                } else {
                    args.extend(amp(&weight.name));
                    args.extend([p(',')]);
                    args.extend(amp_mut(&out.name));
                    // 静态档写进已有输出：`Mat::mul_scaled_into` / `mul_into`
                    // （`apply_into` 在 `Prepared` 上）
                    let name = if scaled {
                        "mul_scaled_into"
                    } else {
                        "mul_into"
                    };
                    body.extend(method(&input.name, name, args));
                }
            }
            // 包进 block：不污染调用点作用域，也让整条公式是一个表达式
            let mut ts = TokenStream::new();
            ts.extend([group(Delimiter::Brace, body)]);
            ts
        }
        // 只写乘积：输出新分配（无下标时无法判档，按全静态生成；DYN 请写下标形式）
        None => {
            if planned {
                let mut args = TokenStream::new();
                alpha_arg(&mut args, f.scalar.as_ref());
                args.extend([amp(&input.name)]);
                let name = match (runtime_rows, scaled) {
                    (true, true) => "apply_scaled_dyn",
                    (true, false) => "apply_dyn",
                    (false, true) => "apply_scaled",
                    (false, false) => "apply",
                };
                method(&weight.name, name, args)
            } else if runtime_rows {
                let mut args = TokenStream::new();
                alpha_arg(&mut args, f.scalar.as_ref());
                args.extend([amp(&input.name)]);
                let name = if scaled {
                    "apply_scaled_dyn"
                } else {
                    "apply_dyn"
                };
                method(&weight.name, name, args)
            } else {
                let mut args = TokenStream::new();
                alpha_arg(&mut args, f.scalar.as_ref());
                args.extend([amp(&weight.name)]);
                let name = if scaled { "mul_scaled" } else { "mul" };
                method(&input.name, name, args)
            }
        }
    }
}

/// 把标量前缀写成**第一个实参**（没有标量就什么都不做）。
fn alpha_arg(args: &mut TokenStream, scalar: Option<&TokenTree>) {
    if let Some(s) = scalar {
        args.extend([s.clone(), p(',')]);
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

/// 向量点积：`dot!(x[K] * y[K])`（`f32`/`f64`/f16 权重·f32 三种 dtype）。
///
/// 接受什么、报错怎么锚，见 crate 文档；实现在 `macros/src/vec.rs`。
#[proc_macro]
pub fn dot(input: TokenStream) -> TokenStream {
    vec::expand_dot(input)
}

/// 矩阵-向量乘：`gemv!(y[N] = w[N, K] * x[K])` / `gemv!(w[N, K] * x[K])`（f16 权重）。
#[proc_macro]
pub fn gemv(input: TokenStream) -> TokenStream {
    vec::expand_gemv(input)
}

/// **`#[fast_trig]`：占位属性，当前是"无操作直通"**。
///
/// **现状（2026-10-16）**：**原样返回 `item`**，不做任何解析/降级/发射 ⇒ **语义逐字不变**。
/// **为什么先立槽位**：战略调整 —— **"识别模式"难、"命名内建"便宜** ⇒ 快速超越函数的正式入口将改为
/// **库侧类型化内建**（形状进类型、用户点名调用、库内 `rewrite::plan()` + `recurrence_safe` 自动选
/// libm/递推）。**本属性降级为将来的糖（可选），不是前置条件。**
///
/// **明确不做**（留待将来）：属性参数解析、表达式形状识别、描述符发射、接重写路径。
#[proc_macro_attribute]
pub fn fast_trig(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
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
        let v = check_names(Some(("M", "N")), ("M", "K"), Some(("Q", "N"))).unwrap_err();
        assert!(matches!(
            v,
            Violation::SameDimDifferentNames { dim: "收缩", .. }
        ));
        // 行维（输出大写、输入小写：档次也不同，提示里会点出来）
        let v = check_names(Some(("M", "N")), ("m", "K"), Some(("K", "N"))).unwrap_err();
        let Violation::SameDimDifferentNames { dim, lhs, rhs, .. } = v;
        assert_eq!(dim, "行");
        assert_eq!((lhs.as_str(), rhs.as_str()), ("M", "m"));
        // 列维
        let v = check_names(Some(("M", "N")), ("M", "K"), Some(("K", "n"))).unwrap_err();
        assert!(matches!(
            v,
            Violation::SameDimDifferentNames { dim: "列", .. }
        ));
        // 正例：全静态、全运行期都合法
        assert!(check_names(Some(("M", "N")), ("M", "K"), Some(("K", "N"))).is_ok());
        assert!(check_names(Some(("m", "N")), ("m", "K"), Some(("K", "N"))).is_ok());
        // 无输出时只查收缩维
        assert!(check_names(None, ("M", "K"), Some(("K", "N"))).is_ok());
        assert!(check_names(None, ("M", "K"), Some(("Q", "N"))).is_err());
    }

    /// plan 形态（权重不带下标）：只查"输出行 == 输入行"，权重侧没有可对照的名字。
    #[test]
    fn plan_form_only_checks_the_row_dimension() {
        // 行维仍然要一致（这就是 plan 形态下唯一的静态维度检查）
        let v = check_names(Some(("M", "N")), ("m", "K"), None).unwrap_err();
        let Violation::SameDimDifferentNames { dim, .. } = v;
        assert_eq!(dim, "行");
        // `matmul!(y[M, N] = x[M, K] * p)` 与 DYN 档都合法
        assert!(check_names(Some(("M", "N")), ("M", "K"), None).is_ok());
        assert!(check_names(Some(("m", "N")), ("m", "K"), None).is_ok());
        // 无输出：plan 形态没有可查的维度，收缩维也查不了
        assert!(check_names(None, ("M", "K"), None).is_ok());
    }

    // 标量前缀（`alpha * …`）的**解析**在这里测不了：`proc_macro` 的 `TokenStream` 一旦
    // 离开真实宏调用就会 panic（"procedural macro API is used outside of a procedural macro"）。
    // 它的正面路径由 `src/shape.rs` 的 `test_formula_dsl_scaled_matches_two_pass_bit_for_bit`
    // 覆盖（宏真的展开、真的跑出逐位一致的结果）；负面路径（诊断文案）与既有做法一致，
    // 靠人工核对的编译输出守着。
}
