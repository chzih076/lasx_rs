//! **`#[fast_trig]` 直通性测试**（2026-10-16）。
//!
//! 属性当前是**无操作直通**（`fn fast_trig(_attr, item) -> TokenStream { item }`）。
//! **为什么用"行为等价"而不是"字符串逐字相同"**：`proc_macro` 的 `TokenStream` **一旦离开真实宏调用
//! 就会 panic**（仓内 `macros/src/lib.rs` 末尾已有此结论），所以**在 crate 内构造 TokenStream 比对
//! 做不到**。⇒ 这里用**下游真实展开**证明：带属性与不带的两个函数**逐位行为相同**；
//! 加上实现本身是"原样返回 `item`"（一行），两条合起来即"直通、不改语义"的证据。

/// 挂了属性：**应当与下面的 `plain` 逐位相同**。
#[lasx_rs_macros::fast_trig]
fn annotated(x: f64) -> f64 {
    x * 2.0 + 1.0
}

/// 不挂属性：对照组。
fn plain(x: f64) -> f64 {
    x * 2.0 + 1.0
}

#[test]
fn passthrough_is_bit_identical() {
    for x in [-3.5f64, -0.0, 0.0, 1.0, 1.25e-7, 1e7, 12345.6789] {
        assert_eq!(
            annotated(x).to_bits(),
            plain(x).to_bits(),
            "带 #[fast_trig] 与不带的函数行为必须逐位相同（直通、不改语义）：x={x}"
        );
    }
}

/// 属性也**不改签名/可见性**：能在模块外被调用（即原 item 原样保留）。
#[test]
fn passthrough_keeps_the_item_usable() {
    fn call_it(f: fn(f64) -> f64) -> f64 {
        f(2.0)
    }
    assert_eq!(call_it(annotated), 5.0);
    assert_eq!(call_it(plain), 5.0);
}
