//! # `pencil_decomp`: Library for MPI pencil distributions
//!
//! ## Documentation
//!
//! <img align="left" src="https://github.com/preiter93/pencil_decomp/blob/master/pics/pencil2.png?raw=true" width="300">
//! <br /> <br /> <br /> <br />
//!
//! ## Notes
//! Work in progress...
#![warn(clippy::pedantic)]
#![warn(missing_docs)]
#[macro_use]
mod internal_macros;

pub mod dist;
pub mod global;
pub mod pencil;
pub use pencil::Pencil;
pub mod decomp3;
pub use decomp3::Decomp3;
pub mod decomp2;
pub use decomp2::Decomp2;
