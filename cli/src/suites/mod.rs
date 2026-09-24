//! 各内核组的基准套件：每个 `pub fn` 对应一个 [`crate::group::Group`]。

pub mod activation;
pub mod align;
pub mod attitude;
pub mod batch;
pub mod large;
pub mod matmul;
pub mod micro;
pub mod physics;
pub mod plan;
pub mod quant;
pub mod reduce;
pub mod rms_norm;
pub mod scenario;
pub mod softmax;
