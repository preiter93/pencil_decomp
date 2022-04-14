//! Pencil decomposition in three dimensions
use crate::pencil::{transpose, Pencil};
use mpi::{environment::Universe, traits::Equivalence};
use ndarray::{ArrayBase, Data, DataMut, Ix3};
use num_traits::Zero;

pub struct Decomp3<'a> {
    /// Mpi universe
    pub universe: &'a Universe,
    /// Total number of grid points [nx global, ny global, nz_global]
    pub n_global: [usize; 3],
    // Size, indices, counts and displacements for x-pencil
    pub x_pencil: Pencil<'a, 3, 2>,
    // Size, indices, counts and displacements for y-pencil
    pub y_pencil: Pencil<'a, 3, 2>,
    // Size, indices, counts and displacements for z-pencil
    pub z_pencil: Pencil<'a, 3, 2>,
}

impl<'a> Decomp3<'a> {
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
        n_global: [usize; 3],
        cart_dims: [i32; 2],
        cart_periodic: [bool; 2],
    ) -> Self {
        let x_pencil = Pencil::new(universe, n_global, 0, cart_dims, cart_periodic);
        let y_pencil = Pencil::new(universe, n_global, 1, cart_dims, cart_periodic);
        let z_pencil = Pencil::new(universe, n_global, 2, cart_dims, cart_periodic);
        Self {
            universe,
            n_global,
            x_pencil,
            y_pencil,
            z_pencil,
        }
    }

    /// Transpose from x to y pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn transpose_x_to_y<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix3>,
        rcv: &mut ArrayBase<S2, Ix3>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.x_pencil.shape());
        check_shape(rcv, self.y_pencil.shape());
        transpose(&self.x_pencil, &self.y_pencil, snd, rcv, split_xy, merge_xy);
    }

    /// Transpose from y to x pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn transpose_y_to_x<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix3>,
        rcv: &mut ArrayBase<S2, Ix3>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.y_pencil.shape());
        check_shape(rcv, self.x_pencil.shape());
        transpose(&self.y_pencil, &self.x_pencil, snd, rcv, split_yx, merge_yx);
    }

    /// Transpose from y to z pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn transpose_y_to_z<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix3>,
        rcv: &mut ArrayBase<S2, Ix3>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.y_pencil.shape());
        check_shape(rcv, self.z_pencil.shape());
        transpose(&self.y_pencil, &self.z_pencil, snd, rcv, split_yz, merge_yz);
    }

    /// Transpose from z to y pencil
    ///
    /// # Panics
    /// Shape mismatch of snd or rcv with send/recv pencil
    pub fn transpose_z_to_y<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix3>,
        rcv: &mut ArrayBase<S2, Ix3>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.z_pencil.shape());
        check_shape(rcv, self.y_pencil.shape());
        transpose(&self.z_pencil, &self.y_pencil, snd, rcv, split_zy, merge_zy);
    }
}

/// Prepare send buffer for `transpose_x_to_y`
fn split_xy<S, T>(
    data: &ArrayBase<S, Ix3>,
    buf: &mut [T],
    x_pencil: &Pencil<3, 2>,
    y_pencil: &Pencil<3, 2>,
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
    data: &ArrayBase<S, Ix3>,
    buf: &mut [T],
    y_pencil: &Pencil<3, 2>,
    x_pencil: &Pencil<3, 2>,
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

/// Prepare send buffer for `transpose_y_to_z`
fn split_yz<S, T>(
    data: &ArrayBase<S, Ix3>,
    buf: &mut [T],
    y_pencil: &Pencil<3, 2>,
    z_pencil: &Pencil<3, 2>,
) where
    S: Data<Elem = T>,
    T: Copy,
{
    assert!(y_pencil.axis_contig == 1);
    assert!(z_pencil.axis_contig == 2);
    let mut data_view = data.view();
    data_view.swap_axes(0, 1);
    for (d, b) in data_view.iter().zip(buf.iter_mut()) {
        *b = *d;
    }
}

/// Prepare send buffer for `transpose_z_to_y`
fn split_zy<S, T>(
    data: &ArrayBase<S, Ix3>,
    buf: &mut [T],
    z_pencil: &Pencil<3, 2>,
    y_pencil: &Pencil<3, 2>,
) where
    S: Data<Elem = T>,
    T: Copy,
{
    assert!(y_pencil.axis_contig == 1);
    assert!(z_pencil.axis_contig == 2);
    let mut data_view = data.view();
    data_view.swap_axes(0, 2);
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
    data: &mut ArrayBase<S, Ix3>,
    x_pencil: &Pencil<3, 2>,
    y_pencil: &Pencil<3, 2>,
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
                for k in 0..y_pencil.dists[2].sz {
                    data[[i, j, k]] = buf[pos];
                    pos += 1;
                }
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
    data: &mut ArrayBase<S, Ix3>,
    y_pencil: &Pencil<3, 2>,
    x_pencil: &Pencil<3, 2>,
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
                for k in 0..x_pencil.dists[2].sz {
                    data[[i, j, k]] = buf[pos];
                    pos += 1;
                }
            }
        }
    }
}

/// Redistribute recv buffer for `transpose_y_to_z`
///
/// # Panics
/// i32 to usize conversion fails
fn merge_yz<S, T>(
    buf: &[T],
    data: &mut ArrayBase<S, Ix3>,
    y_pencil: &Pencil<3, 2>,
    z_pencil: &Pencil<3, 2>,
) where
    S: DataMut<Elem = T>,
    T: Copy,
{
    assert!(y_pencil.axis_contig == 1);
    assert!(z_pencil.axis_contig == 2);
    let mut pos = 0;
    let nprocs = y_pencil.nprocs_along_axis(2);
    for proc in 0..nprocs.try_into().unwrap() {
        let k1 = y_pencil.dists[2].st_procs[proc];
        let k2 = y_pencil.dists[2].en_procs[proc];
        for j in 0..z_pencil.dists[1].sz {
            for i in 0..z_pencil.dists[0].sz {
                for k in k1..=k2 {
                    data[[i, j, k]] = buf[pos];
                    pos += 1;
                }
            }
        }
    }
}

/// Redistribute recv buffer for `transpose_z_to_y`
///
/// # Panics
/// i32 to usize conversion fails
fn merge_zy<S, T>(
    buf: &[T],
    data: &mut ArrayBase<S, Ix3>,
    z_pencil: &Pencil<3, 2>,
    y_pencil: &Pencil<3, 2>,
) where
    S: DataMut<Elem = T>,
    T: Copy,
{
    assert!(y_pencil.axis_contig == 1);
    assert!(z_pencil.axis_contig == 2);
    let mut pos = 0;
    let nprocs = z_pencil.nprocs_along_axis(1);
    for proc in 0..nprocs.try_into().unwrap() {
        let j1 = z_pencil.dists[1].st_procs[proc];
        let j2 = z_pencil.dists[1].en_procs[proc];
        for k in 0..y_pencil.dists[2].sz {
            for j in j1..=j2 {
                for i in 0..y_pencil.dists[0].sz {
                    data[[i, j, k]] = buf[pos];
                    pos += 1;
                }
            }
        }
    }
}

/// # Panics
/// Panics if array shape does not conform with pencil distribution
fn check_shape<A, S, const N: usize>(
    data: &ndarray::ArrayBase<S, ndarray::Dim<[usize; N]>>,
    shape: [usize; N],
) where
    S: ndarray::Data<Elem = A>,
    ndarray::Dim<[usize; N]>: ndarray::Dimension,
{
    assert!(
        !(data.shape() != shape),
        "Shape mismatch, got {:?} expected {:?}.",
        data.shape(),
        shape
    );
}
