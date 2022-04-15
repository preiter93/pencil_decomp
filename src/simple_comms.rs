//! Collection of simple global mpi routines
use mpi::collective::CommunicatorCollectives;
use mpi::collective::Root;
use mpi::environment::Universe;
use mpi::topology::Communicator;
use mpi::traits::Equivalence;
use num_traits::Zero;

/// Broadcast scalar value from root to all processes
pub fn broadcast_scalar<T: Zero + Equivalence>(universe: &Universe, data: &mut T) {
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    root_process.broadcast_into(data);
}

/// Gather values on root and apply a closure function
///
/// See also [`gather_sum`]
///
/// # Panics
/// i32 to usize conversion
pub fn gather_apply<T, F>(universe: &Universe, data: &T, result: &mut T, f: F)
where
    T: Zero + Equivalence + Clone,
    F: Fn(&[T]) -> T,
{
    let world = universe.world();
    let size = world.size().try_into().unwrap();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    if world.rank() == root_rank {
        let mut a = vec![T::zero(); size];
        root_process.gather_into_root(data, &mut a[..]);
        *result = f(&a);
    } else {
        root_process.gather_into(data);
    }
}

/// Gather sum of values on root
pub fn gather_sum<T>(universe: &Universe, data: &T, result: &mut T)
where
    T: Zero + Equivalence + Clone + Copy + std::iter::Sum,
{
    let f = |x: &[T]| x.iter().copied().sum();
    gather_apply(universe, data, result, f);
}

/// Gather values on all processes and apply a closure function
///
/// See also [`all_gather_sum`]
///
/// # Panics
/// i32 to usize conversion
pub fn all_gather_apply<T, F>(universe: &Universe, data: &T, result: &mut T, f: F)
where
    T: Zero + Equivalence + Clone,
    F: Fn(&[T]) -> T,
{
    let world = universe.world();
    let size = world.size().try_into().unwrap();
    let mut a = vec![T::zero(); size];
    world.all_gather_into(data, &mut a[..]);
    *result = f(&a);
}

/// Gather sum of values on all processes
pub fn all_gather_sum<T>(universe: &Universe, data: &T, result: &mut T)
where
    T: Zero + Equivalence + Clone + Copy + std::iter::Sum,
{
    let f = |x: &[T]| x.iter().copied().sum();
    all_gather_apply(universe, data, result, f);
}
