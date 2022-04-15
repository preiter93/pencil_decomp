//! # Pencil distributed data
use crate::distribution::Distribution;
use mpi::topology::Communicator;
use mpi::{
    collective::CommunicatorCollectives, collective::Root, datatype::Partition,
    datatype::PartitionMut, environment::Universe, topology::CartesianCommunicator,
    topology::CartesianLayout, traits::Equivalence, Count,
};
use num_traits::Zero;

/// Pencil Distribution
///
/// *M* number of grid dimensions.
/// *N* specifies number of dimension of the cartesian topology,
/// Currently restricted to *N* = *M* - 1
pub struct Pencil<'a, const M: usize, const N: usize> {
    /// Mpi universe
    pub universe: &'a Universe,
    /// Communicator
    pub comm: CartesianCommunicator,
    /// Grid point distribution along each axis
    pub dists: [Distribution; M],
    /// One axis is contiguous
    pub axis_contig: usize,
}

impl<'a, const M: usize, const N: usize> Pencil<'a, M, N> {
    /// Construct pencil distribution
    ///
    /// # Arguments
    /// * `universe`     : Mpi Universe
    /// * `n_global`     : Total number of grid points [nx global, ny global]
    /// * `axis_contig`  : Contiguous axis
    /// * `cart_ndims`   : Number of dimensions of cartesian grid
    /// * `cart_periodic`: Logical array of size ``cart_ndims`` specifying whether the grid is periodic
    ///
    /// # Panics
    /// - Mismatch of *ndims* and number of processors
    ///
    /// # Example
    /// Basic usage of Pencil. Run with 2 processors, i.e.
    /// ``cargo mpirun --np 2 ...``
    /// ```ignore
    /// use mpi::topology::Communicator;
    /// use pencil_decomp::Pencil;
    /// // Init Mpi
    /// let universe = mpi::initialize().unwrap();
    /// let world = universe.world();
    /// assert!(world.size() == 6, "Run with 6 processors");
    ///
    /// // Parameters
    /// let n_global = [6, 4];
    /// let cart_dims = [2];
    /// let cart_periodic = [false];
    /// let axis_contig = 0;
    ///  
    /// // Pencil split along y-axis
    /// let pencil = Pencil::new(&universe, n_global, axis_contig, cart_dims, cart_periodic);
    /// assert_eq!(pencil.len(), 6 * 2);
    /// ```
    #[must_use]
    pub fn new(
        universe: &'a Universe,
        n_global: [usize; M],
        axis_contig: usize,
        cart_ndims: [i32; N],
        cart_periodic: [bool; N],
    ) -> Self {
        // Contiguous axis must be < M
        assert!(axis_contig < M, "Contiguous axis must be < M");
        // Dim of cartesian topology should be one less than dim of grid
        assert!(
            N == M - 1,
            "Dimensionality mismatch, expect N == M - 1, check cart_ndims"
        );
        // Mpi comm world
        let world_comm = universe.world();
        // Check number of processors
        let n = cart_ndims.iter().product::<i32>();
        let m = world_comm.size();
        let cn = cart_ndims;
        assert!(n == m, "Expect {} procs for grid {:?}, got {}", n, cn, m);
        // Create cartesian communicator
        let comm = world_comm
            .create_cartesian_communicator(&cart_ndims, &cart_periodic, false)
            .unwrap();
        // Distribute grid points
        let mut dists: Vec<Distribution> = Vec::new();
        // nrank
        let CartesianLayout { coords, .. } = comm.get_layout();
        let mut dim = 0;
        for (i, &n_dim) in n_global.iter().enumerate() {
            if i == axis_contig {
                dists.push(Distribution::contiguous(n_dim));
            } else {
                dists.push(Distribution::split(
                    n_dim,
                    cart_ndims[dim].try_into().unwrap(),
                    coords[dim].try_into().unwrap(),
                ));
                dim += 1;
            }
        }
        // Convert to array
        let dists: [Distribution; M] = dists.try_into().unwrap();
        Self {
            universe,
            comm,
            // n_global,
            dists,
            axis_contig,
        }
    }

    /// Gets the coordinate of a process in a communicator that has a cartesian topology.
    ///
    /// # Panics
    /// Must be of size *N*
    #[must_use]
    pub fn cart_coords(&self) -> Vec<i32> {
        let CartesianLayout { coords, .. } = self.comm.get_layout();
        assert!(coords.len() == N);
        coords
    }

    /// Gets integer array of size ndims specifying the number of
    /// processes in each dimension
    ///
    /// # Panics
    /// Int conversion
    #[must_use]
    pub fn cart_dims(&self) -> Vec<i32> {
        let CartesianLayout { dims, .. } = self.comm.get_layout();
        assert!(dims.len() == N);
        dims
    }

    /// Gets logical array of size ndims specifying whether the grid is periodic
    ///
    /// # Panics
    /// Int conversion
    #[must_use]
    pub fn cart_periodic(&self) -> Vec<bool> {
        let CartesianLayout { periods, .. } = self.comm.get_layout();
        assert!(periods.len() == N);
        periods
    }

    /// Maps physical dimension to cartesian topology dimension
    ///
    /// For example, if contiguos axis is 1, then
    /// ``dim`` = 0 -> ``cart_dim`` = 0,
    /// ``dim`` = 2 -> ``cart_dim`` = 1.
    ///
    /// # Panics
    /// - If *dim* equals contiguos axis
    /// - If *dim* is larger than *M*
    fn map_dim_to_cart_dim(&self, dim: usize) -> usize {
        assert!(dim < M);
        match dim.cmp(&self.axis_contig) {
            std::cmp::Ordering::Less => dim,
            std::cmp::Ordering::Greater => dim - 1,
            std::cmp::Ordering::Equal => panic!("dim must differ from axis_contig"),
        }
    }

    /// Return number of splits / processors along a certain dimension(axis)
    #[must_use]
    pub fn nprocs_along_axis(&self, axis: usize) -> i32 {
        let cart_dim = self.map_dim_to_cart_dim(axis);
        self.cart_dims()[cart_dim]
    }

    /// Return communicator defining sub-groups for ALLTOALL(V)
    ///
    /// # Panics
    /// *dim* must be different from contiguous axis, cartesian
    /// communicator only communicates between split dimensions
    #[must_use]
    pub fn subcomm_along_axis(&self, axis: usize) -> CartesianCommunicator {
        let cart_dim = self.map_dim_to_cart_dim(axis);
        let mut retain = [false; N];
        retain[cart_dim] = true;
        self.comm.subgroup(&retain)
    }

    /// Return the total length of data hold by current processor
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Return the total length of data hold by current processor
    #[must_use]
    pub fn len_global(&self) -> usize {
        self.shape_global().iter().product()
    }

    /// Returns true if self has a length of zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape().iter().product::<usize>() == 0
    }

    /// Shape of pencil distributed data
    ///
    /// # Panics
    /// Vector to array conversion fails
    #[must_use]
    pub fn shape(&self) -> [usize; M] {
        self.dists
            .iter()
            .map(|x| x.sz)
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    }

    /// Shape of global data
    ///
    /// # Panics
    /// Vector to array conversion fails
    #[must_use]
    pub fn shape_global(&self) -> [usize; M] {
        self.dists
            .iter()
            .map(|x| x.sz_procs.iter().sum())
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    }
}

/// Transpose between pencils
///
/// See for example [`pencil_decomp::decomp3::transpose_x_to_y`]
pub(crate) fn transpose<S, R, T, Split, Merge, const M: usize, const N: usize>(
    send_pencil: &Pencil<M, N>,
    recv_pencil: &Pencil<M, N>,
    snd: &S,
    rcv: &mut R,
    split: Split,
    merge: Merge,
) where
    T: Zero + Copy + Equivalence,
    Split: Fn(&S, &mut [T], &Pencil<M, N>, &Pencil<M, N>),
    Merge: Fn(&[T], &mut R, &Pencil<M, N>, &Pencil<M, N>),
{
    assert!(send_pencil.axis_contig != recv_pencil.axis_contig);

    // send & receive buffer
    let mut send_buf = vec![T::zero(); send_pencil.len()];
    let mut recv_buf = vec![T::zero(); recv_pencil.len()];
    split(snd, &mut send_buf, send_pencil, recv_pencil);

    let (send_counts, send_displs) = send_counts_all_to_all(send_pencil, recv_pencil);
    let (recv_counts, recv_displs) = recv_counts_all_to_all(send_pencil, recv_pencil);
    let comm = send_pencil.subcomm_along_axis(recv_pencil.axis_contig);
    {
        let send_buffer = Partition::new(&send_buf[..], &send_counts[..], &send_displs[..]);
        let mut recv_buffer =
            PartitionMut::new(&mut recv_buf[..], &recv_counts[..], &recv_displs[..]);
        comm.all_to_all_varcount_into(&send_buffer, &mut recv_buffer);
    }

    // copy receive buffer into array
    merge(&recv_buf, rcv, send_pencil, recv_pencil);
}

/// Gather pencil along axis into root
///
/// See for example [`pencil_decomp::decomp2::gather_x`]
pub(crate) fn gather_into_root_along_axis<S, R, T, Split, Merge, const M: usize, const N: usize>(
    pencil: &Pencil<M, N>,
    snd: &S,
    rcv: &mut R,
    axis: usize,
    split: Split,
    merge: Merge,
) where
    T: Zero + Copy + Equivalence,
    Split: Fn(&S, &mut [T]),
    Merge: Fn(&[T], &mut R),
{
    // Asserts are handled in `recv_counts_gather_axis`

    let root_rank = 0;
    let comm = pencil.subcomm_along_axis(axis);
    let root_process = comm.process_at_rank(root_rank);

    let mut send_buf = vec![T::zero(); pencil.len()];
    split(snd, &mut send_buf);
    if comm.rank() == root_rank {
        let mut recv_buf = vec![T::zero(); pencil.len_global()];

        let (counts, displs) = recv_counts_gather_axis(pencil, axis);
        {
            let mut partition = PartitionMut::new(&mut recv_buf[..], &counts[..], &displs[..]);
            root_process.gather_varcount_into_root(&send_buf[..], &mut partition);
        }
        // copy receive buffer into array
        merge(&recv_buf, rcv);
    } else {
        root_process.gather_varcount_into(&send_buf[..]);
    }
}

/// Gather pencil along axis into root
///
/// See for example [`pencil_decomp::decomp2::gather_x`]
pub(crate) fn scatter_along_axis<S, R, T, Split, Merge, const M: usize, const N: usize>(
    pencil: &Pencil<M, N>,
    snd: &S,
    rcv: &mut R,
    axis: usize,
    split: Split,
    merge: Merge,
) where
    T: Zero + Copy + Equivalence,
    Split: Fn(&S, &mut [T]),
    Merge: Fn(&[T], &mut R),
{
    // Asserts are handled in sub-functions

    let root_rank = 0;
    let comm = pencil.subcomm_along_axis(axis);
    let root_process = comm.process_at_rank(root_rank);

    // recv buffer
    let mut recv_buf = vec![T::zero(); pencil.len()];

    if comm.rank() == root_rank {
        // send buffer
        let mut send_buf = vec![T::zero(); pencil.len_global()];
        split(snd, &mut send_buf);

        let (counts, displs) = recv_counts_gather_axis(pencil, axis);
        {
            let partition = Partition::new(&send_buf[..], &counts[..], &displs[..]);
            root_process.scatter_varcount_into_root(&partition, &mut recv_buf[..]);
        }
    } else {
        root_process.scatter_varcount_into(&mut recv_buf[..]);
    }
    // copy receive buffer into array
    merge(&recv_buf, rcv);
}

/// Returns send counts and displs from two ``Pencil`` for
/// mpis ``mpi_all_to_allv`` routine
///
/// To get recv counts and displs, reverse order of send & recv
/// pencil, see [`recv_counts_all_to_all`]
///
/// # Panics
/// - send and recv pencil must not have same contiguous axis
/// - transpose cant be done with ``all_to_all_v``, use ``all_to_all_w`` instead (not implemented)
/// - i32 to usize conversion fails
#[must_use]
pub fn send_counts_all_to_all<const M: usize, const N: usize>(
    send: &Pencil<M, N>,
    recv: &Pencil<M, N>,
) -> (Vec<Count>, Vec<Count>) {
    assert!(
        send.axis_contig != recv.axis_contig,
        "Expect pencils with different contiguous axes."
    );
    let nprocs = send.nprocs_along_axis(recv.axis_contig);
    // counts
    let counts = (0..nprocs.try_into().unwrap())
        .map(|np| {
            let mut count = 1;
            for i in 0..M {
                if i != send.axis_contig && i != recv.axis_contig {
                    // Both pencils are split.
                    // Check if they are split in the same way,
                    // otherwise ``all_to_all_v`` wont work
                    assert!(
                        send.dists[i].sz == recv.dists[i].sz,
                        "unable to get send counts. Maybe you need to use manual all_to_all_w."
                    );
                    //count *= send.dists[i].sz_procs[np];
                    count *= send.dists[i].sz;
                } else if i == recv.axis_contig {
                    // Send is split, recv is contiguous
                    // count *= send.dists[i].sz_procs[np];
                    count *= send.dists[i].sz;
                } else {
                    // Recv is split, send is contiguous
                    count *= recv.dists[i].sz_procs[np];
                }
            }
            count.try_into().unwrap()
        })
        .collect::<Vec<Count>>();

    // displacements
    let displs: Vec<Count> = counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();
    (counts, displs)
}

/// Returns recv counts and displs from two ``Pencil``
///
/// Just calls [`send_counts_all_to_all`] with reversed
/// send/recv order
#[must_use]
pub fn recv_counts_all_to_all<const M: usize, const N: usize>(
    send: &Pencil<M, N>,
    recv: &Pencil<M, N>,
) -> (Vec<Count>, Vec<Count>) {
    send_counts_all_to_all(recv, send)
}

/// Returns recv counts and displs from a ``Pencil`` to merge
/// a given axis. This must be used in combination with ``mpi_gather``
///
/// # Panics
/// - axis is already merged (``axis == axis_contig ``)
///
/// # Example
/// Run with 2 processors
/// ```ìgnore
/// use mpi::topology::Communicator;
/// use pencil_decomp::pencil::{Pencil, recv_counts_gather_axis};
///
/// // Init Mpi
/// let universe = mpi::initialize().unwrap();
/// let world = universe.world();
/// assert!(world.size() == 2, "Run with 2 processors");
///
/// // Parameters
/// let n_global = [6, 5];
/// let cart_dims = [2];
/// let cart_periodic = [false];
/// let axis_contig = 0;
///
/// let pencil = Pencil::new(&universe, n_global, axis_contig, cart_dims, cart_periodic);
///
/// let (counts, displs) = recv_counts_gather_axis(&pencil, 1);
/// if world.rank() == 0 {
///     assert_eq!(counts, [12, 18]);
///     assert_eq!(displs, [0, 12]);
/// }
/// ```
#[must_use]
pub fn recv_counts_gather_axis<const M: usize, const N: usize>(
    pencil: &Pencil<M, N>,
    axis: usize,
) -> (Vec<Count>, Vec<Count>) {
    assert!(pencil.axis_contig != axis, "Axis {} is already cont.", axis);
    assert!(axis < M, "Axis {} outside array dimensions {}.", M, axis);

    // Number of procs along dimension
    let nprocs = pencil.nprocs_along_axis(axis);

    let counts = (0..nprocs.try_into().unwrap())
        .map(|np| {
            let mut count = 1;
            for i in 0..M {
                if i == axis {
                    count *= pencil.dists[i].sz_procs[np];
                } else {
                    count *= pencil.dists[i].sz;
                }
            }
            count.try_into().unwrap()
        })
        .collect::<Vec<Count>>();

    let displs: Vec<Count> = counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    (counts, displs)
}
