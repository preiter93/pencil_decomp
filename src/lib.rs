//! # `pencil_decomp`: Library for MPI pencil distributions
//!
//! ## Documentation
//! <div align="left">
//! <img src="https://github.com/preiter93/pencil_decomp/blob/master/pics/pencil2.png?raw=true" width="500"></img>
//! </div>
//!
//! ![Alt version](https://github.com/preiter93/pencil_decomp/blob/master/pics/pencil2.png)
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
