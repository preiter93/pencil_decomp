//! Pencil decomposition in two dimensions
use crate::pencil::{gather_into_root_along_axis, scatter_along_axis, transpose, Pencil};
use mpi::{environment::Universe, traits::Equivalence};
use ndarray::{ArrayBase, Data, DataMut, Ix2};
use num_traits::Zero;

/// Pencil decomposition in three dimensions
pub struct Decomp2<'a> {
    /// Mpi universe
    pub universe: &'a Universe,
    /// Total number of grid points [nx global, ny global]
    pub n_global: [usize; 2],
    /// Size, indices, counts and displacements for x-pencil
    pub x_pencil: Pencil<'a, 2, 1>,
    /// Size, indices, counts and displacements for y-pencil
    pub y_pencil: Pencil<'a, 2, 1>,
}

impl<'a> Decomp2<'a> {
    /// Construct pencil distribution
    ///
    /// # Arguments
    /// * `universe`     : Mpi Universe
    /// * `n_global`     : Total number of grid points [nx global, ny global]
    /// * `cart_ndims`   : Number of dimensions of cartesian grid
    /// * `cart_periodic`: Logical array of size ``cart_ndims`` specifying whether the grid is periodic
    ///
    /// # Panics
    /// - Mismatch of *ndims* and number of processors
    #[must_use]
    pub fn new(
        universe: &'a Universe,
        n_global: [usize; 2],
        cart_dims: [i32; 1],
        cart_periodic: [bool; 1],
    ) -> Self {
        let x_pencil = Pencil::new(universe, n_global, 0, cart_dims, cart_periodic);
        let y_pencil = Pencil::new(universe, n_global, 1, cart_dims, cart_periodic);
        Self {
            universe,
            n_global,
            x_pencil,
            y_pencil,
        }
    }

    /// Transpose from x to y pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn transpose_x_to_y<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        assert_eq_shape!(snd, self.x_pencil, "transpose_x_to_y");
        assert_eq_shape!(rcv, self.y_pencil, "transpose_x_to_y");
        transpose(&self.x_pencil, &self.y_pencil, snd, rcv, split_xy, merge_xy);
    }

    /// Transpose from y to x pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn transpose_y_to_x<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        assert_eq_shape!(snd, self.y_pencil, "transpose_y_to_x");
        assert_eq_shape!(rcv, self.x_pencil, "transpose_y_to_x");
        transpose(&self.y_pencil, &self.x_pencil, snd, rcv, split_yx, merge_yx);
    }

    /// Gather data from x-pencil to root processor
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn gather_x<S1, S2, T>(&self, snd: &ArrayBase<S1, Ix2>, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        assert_eq_shape!(snd, self.x_pencil, "gather_x");
        assert_eq!(rcv.shape(), self.n_global);

        gather_into_root_along_axis(&self.x_pencil, snd, rcv, 1, split_gather_x, merge_gather_x);
    }

    /// Gather data from y-pencil to root processor
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn gather_y<S1, S2, T>(&self, snd: &ArrayBase<S1, Ix2>, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        assert_eq_shape!(snd, self.y_pencil, "gather_y");
        assert_eq!(rcv.shape(), self.n_global);

        gather_into_root_along_axis(&self.y_pencil, snd, rcv, 0, split_gather_y, merge_gather_y);
    }

    /// Scatter data from root to x-pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn scatter_x<S1, S2, T>(&self, snd: &ArrayBase<S1, Ix2>, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        assert_eq_shape!(rcv, self.x_pencil, "gather_x");
        assert_eq!(snd.shape(), self.n_global);

        scatter_along_axis(&self.x_pencil, snd, rcv, 1, split_gather_x, merge_gather_x);
    }

    /// Scatter data from root to y-pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn scatter_y<S1, S2, T>(&self, snd: &ArrayBase<S1, Ix2>, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        assert_eq_shape!(rcv, self.y_pencil, "gather_y");
        assert_eq!(snd.shape(), self.n_global);

        scatter_along_axis(&self.y_pencil, snd, rcv, 0, split_gather_y, merge_gather_y);
    }
}

/// Prepare send buffer for `transpose_x_to_y`
fn split_xy<S, T>(
    data: &ArrayBase<S, Ix2>,
    buf: &mut [T],
    x_pencil: &Pencil<2, 1>,
    y_pencil: &Pencil<2, 1>,
) where
    S: Data<Elem = T>,
    T: Copy,
{
    assert!(x_pencil.axis_contig == 0);
    assert!(y_pencil.axis_contig == 1);
    for (d, b) in data.iter().zip(buf.iter_mut()) {
        *b = *d;
    }
}

/// Prepare send buffer for `transpose_y_to_x`
fn split_yx<S, T>(
    data: &ArrayBase<S, Ix2>,
    buf: &mut [T],
    y_pencil: &Pencil<2, 1>,
    x_pencil: &Pencil<2, 1>,
) where
    S: Data<Elem = T>,
    T: Copy,
{
    assert!(x_pencil.axis_contig == 0);
    assert!(y_pencil.axis_contig == 1);
    let mut data_view = data.view();
    data_view.swap_axes(0, 1);
    for (d, b) in data_view.iter().zip(buf.iter_mut()) {
        *b = *d;
    }
}

/// Redistribute recv buffer for `transpose_x_to_y`
///
/// # Panics
/// i32 to usize conversion fails
fn merge_xy<S, T>(
    buf: &[T],
    data: &mut ArrayBase<S, Ix2>,
    x_pencil: &Pencil<2, 1>,
    y_pencil: &Pencil<2, 1>,
) where
    S: DataMut<Elem = T>,
    T: Copy,
{
    assert!(x_pencil.axis_contig == 0);
    assert!(y_pencil.axis_contig == 1);
    let mut pos = 0;
    let nprocs = x_pencil.nprocs_along_axis(1);
    for proc in 0..nprocs.try_into().unwrap() {
        let j1 = x_pencil.dists[1].st_procs[proc];
        let j2 = x_pencil.dists[1].en_procs[proc];
        for i in 0..y_pencil.dists[0].sz {
            for j in j1..=j2 {
                data[[i, j]] = buf[pos];
                pos += 1;
            }
        }
    }
}

/// Redistribute recv buffer for `transpose_y_to_x`
///
/// # Panics
/// i32 to usize conversion fails
fn merge_yx<S, T>(
    buf: &[T],
    data: &mut ArrayBase<S, Ix2>,
    y_pencil: &Pencil<2, 1>,
    x_pencil: &Pencil<2, 1>,
) where
    S: DataMut<Elem = T>,
    T: Copy,
{
    assert!(x_pencil.axis_contig == 0);
    assert!(y_pencil.axis_contig == 1);
    let mut pos = 0;
    let nprocs = y_pencil.nprocs_along_axis(0);
    for proc in 0..nprocs.try_into().unwrap() {
        let i1 = y_pencil.dists[0].st_procs[proc];
        let i2 = y_pencil.dists[0].en_procs[proc];
        for j in 0..x_pencil.dists[1].sz {
            for i in i1..=i2 {
                data[[i, j]] = buf[pos];
                pos += 1;
            }
        }
    }
}

/// Split for `gather_x`
fn split_gather_x<S: Data<Elem = T>, T: Copy>(data: &ArrayBase<S, Ix2>, buf: &mut [T]) {
    let mut data_view = data.view();
    data_view.swap_axes(0, 1);
    for (d, b) in data_view.iter().zip(buf.iter_mut()) {
        *b = *d;
    }
}

// Split for `gather_y`
fn split_gather_y<S: Data<Elem = T>, T: Copy>(data: &ArrayBase<S, Ix2>, buf: &mut [T]) {
    for (d, b) in data.iter().zip(buf.iter_mut()) {
        *b = *d;
    }
}

// Merge for `gather_x`
fn merge_gather_x<S: DataMut<Elem = T>, T: Copy>(buf: &[T], data: &mut ArrayBase<S, Ix2>) {
    data.swap_axes(0, 1);
    for (d, b) in data.iter_mut().zip(buf.iter()) {
        *d = *b;
    }
    data.swap_axes(0, 1);
}

// Merge
fn merge_gather_y<S: DataMut<Elem = T>, T: Copy>(buf: &[T], data: &mut ArrayBase<S, Ix2>) {
    for (d, b) in data.iter_mut().zip(buf.iter()) {
        *d = *b;
    }
}
