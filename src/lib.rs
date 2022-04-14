//! # `pencil_decomp`: Library for MPI pencil distributions
//!
//! # Explanation
//! ['Pencil decomposition in 3D'](https://github.com/preiter93/pencil_decomp/tree/master/pics/pencil2.pdf)
//!
//! Work in progress...
#![warn(clippy::pedantic)]
pub mod dist;
pub mod global;
pub mod pencil;
pub use pencil::Pencil;
pub mod decomp3;
pub use decomp3::Decomp3;
