//! # `pencil_decomp`: Library for MPI pencil distributions
//!
//! ## Documentation
//!
//! <img align="left" src="https://github.com/preiter93/pencil_decomp/blob/master/pics/pencil2.png?raw=true" width="200"></img>
//!
//! ## Notes
//! Work in progress...
#![warn(clippy::pedantic)]
pub mod dist;
pub mod global;
pub mod pencil;
pub use pencil::Pencil;
pub mod decomp3;
pub use decomp3::Decomp3;
