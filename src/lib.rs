//! # `mpi_pencil`: Library for mpi pencil distributions
//!
//! Work in progress...
#![warn(clippy::pedantic)]
pub mod dist;
pub mod global;
pub mod pencil;
pub use pencil::Pencil;
pub mod decomp3;
pub use decomp3::Decomp3;
