//! cargo mpirun --n 4 --bin pencil_decomp
use mpi::topology::Communicator;
use mpi::{collective::CommunicatorCollectives, datatype::Partition, datatype::PartitionMut};
use mpi::{traits::Equivalence, Count};
use ndarray::{Array3, ArrayBase, Data, DataMut, Ix3};
use num_traits::Zero;
use pencil_decomp::pencil::send_counts_all_to_all;
use pencil_decomp::Pencil;

fn test_array_from_pencil(pencil: &Pencil<3, 2>) -> Array3<f64> {
    let mut data: Array3<f64> = Array3::zeros(pencil.shape());
    for i in pencil.dists[0].st..=pencil.dists[0].en {
        let ii = i - pencil.dists[0].st;
        for j in pencil.dists[1].st..=pencil.dists[1].en {
            let jj = j - pencil.dists[1].st;
            for k in pencil.dists[2].st..=pencil.dists[2].en {
                let kk = k - pencil.dists[2].st;
                data[[ii, jj, kk]] = (i + j * 10 + k * 100) as f64;
            }
        }
    }
    data
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    assert!(world.size() == 4, "Run with 4 processors");
    let n_global = [6, 4, 5];

    let dims = [2, 2];
    let periodic = [false, false];
    let x_pencil = Pencil::new(&universe, n_global, 0, dims, periodic);
    let y_pencil = Pencil::new(&universe, n_global, 1, dims, periodic);

    // transpose x to y
    let x_data: Array3<f64> = test_array_from_pencil(&x_pencil);
    let mut y_data: Array3<f64> = Array3::zeros(y_pencil.shape());
    transpose_x_to_y(&x_pencil, &y_pencil, &x_data, &mut y_data);
    assert_eq!(y_data, test_array_from_pencil(&y_pencil));
    if rank == 0 {
        println!("{:?}", y_data);
    }
    println!("{:?}", x_pencil.shape_global());
}

/// Transpose from x to y pencil
pub fn transpose_x_to_y<S1, S2, T>(
    x_pencil: &Pencil<3, 2>,
    y_pencil: &Pencil<3, 2>,
    snd: &ArrayBase<S1, Ix3>,
    rcv: &mut ArrayBase<S2, Ix3>,
) where
    S1: Data<Elem = T>,
    S2: DataMut<Elem = T>,
    T: Zero + Copy + Equivalence,
{
    assert!(x_pencil.axis_contig == 0);
    assert!(y_pencil.axis_contig == 1);
    assert_eq!(snd.shape(), x_pencil.shape());
    assert_eq!(rcv.shape(), y_pencil.shape());

    // send & receive buffer
    let mut send = vec![T::zero(); x_pencil.length()];
    let mut recv = vec![T::zero(); y_pencil.length()];
    split_xy(snd, &mut send);

    let (send_counts, send_displs) = send_counts_all_to_all(x_pencil, y_pencil);
    let (recv_counts, recv_displs) = send_counts_all_to_all(y_pencil, x_pencil);

    let comm = x_pencil.subcomm_along_axis(1);
    {
        let send_buffer = Partition::new(&send[..], &send_counts[..], &send_displs[..]);
        let mut recv_buffer = PartitionMut::new(&mut recv[..], &recv_counts[..], &recv_displs[..]);
        comm.all_to_all_varcount_into(&send_buffer, &mut recv_buffer);
    }

    // copy receive buffer into array
    // *todo*: prevent copy by referencing?
    merge_xy(x_pencil, y_pencil, rcv, &recv);
}

/// Prepare send buffer for `transpose_x_to_y`
fn split_xy<S, T>(data: &ArrayBase<S, Ix3>, buf: &mut [T])
where
    S: Data<Elem = T>,
    T: Copy,
{
    for (d, b) in data.iter().zip(buf.iter_mut()) {
        *b = *d;
    }
}

/// Redistribute recv buffer for `transpose_x_to_y`
fn merge_xy<S, T>(
    x_pencil: &Pencil<3, 2>,
    y_pencil: &Pencil<3, 2>,
    data: &mut ArrayBase<S, Ix3>,
    buf: &[T],
) where
    S: DataMut<Elem = T>,
    T: Copy,
{
    assert!(x_pencil.axis_contig == 0);
    assert!(y_pencil.axis_contig == 1);
    let mut pos = 0;
    let nprocs = x_pencil.nprocs_along_axis(1);
    for proc in 0..nprocs as usize {
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
//
// /// Returns send counts and displs
// ///
// /// To get recv counts and displs, reverse order of send & recv
// /// pencil
// ///
// /// # Panics
// /// - send and recv pencil must not have same contiguous axis
// fn send_counts_all_to_all<const M: usize, const N: usize>(
//     send: &Pencil<M, N>,
//     recv: &Pencil<M, N>,
// ) -> (Vec<Count>, Vec<Count>) {
//     assert!(
//         send.axis_contig != recv.axis_contig,
//         "Expect pencils with different contiguous axes."
//     );
//     let nprocs = send.nprocs_along_axis(recv.axis_contig);
//     // counts
//     let counts = (0..nprocs)
//         .map(|_| {
//             let mut count = 1;
//             for i in 0..M {
//                 if i == recv.axis_contig {
//                     count *= send.dists[i].sz
//                 } else {
//                     count *= recv.dists[i].sz
//                 }
//             }
//             count.try_into().unwrap()
//         })
//         .collect::<Vec<Count>>();
//
//     // displacements
//     let displs: Vec<Count> = counts
//         .iter()
//         .scan(0, |acc, &x| {
//             let tmp = *acc;
//             *acc += x;
//             Some(tmp)
//         })
//         .collect();
//     (counts, displs)
// }
