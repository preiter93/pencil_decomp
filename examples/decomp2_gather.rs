//! cargo mpirun --np 2 --example decomp2_gather
use mpi::topology::Communicator;
use ndarray::Array2;
use pencil_decomp::pencil::recv_counts_gather_axis;
use pencil_decomp::{Decomp2, Pencil};

fn main() {
    // Init Mpi
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    assert!(world.size() == 2, "Run with 2 processors");

    // Parameters
    let n_global = [6, 5];
    let cart_dims = [2];
    let cart_periodic = [false];

    let decomp2 = Decomp2::new(&universe, n_global, cart_dims, cart_periodic);

    let (counts, displs) = recv_counts_gather_axis(&decomp2.x_pencil, 1);
    if world.rank() == 0 {
        assert_eq!(counts, [12, 18]);
        assert_eq!(displs, [0, 12]);
    }

    // Gather x
    let x_data = test_array_from_pencil(&decomp2.x_pencil);
    let mut data = Array2::zeros(n_global);
    decomp2.gather_x(&x_data, &mut data);

    if world.rank() == 0 {
        assert_eq!(data, test_array(n_global, [0, 0]));
    }

    // Gather y
    let y_data = test_array_from_pencil(&decomp2.y_pencil);
    let mut data = Array2::zeros(n_global);
    decomp2.gather_y(&y_data, &mut data);

    if world.rank() == 0 {
        assert_eq!(data, test_array(n_global, [0, 0]));
    }
}

fn test_array_from_pencil(pencil: &Pencil<2, 1>) -> Array2<f64> {
    test_array(pencil.shape(), [pencil.dists[0].st, pencil.dists[1].st])
}

fn test_array(shape: [usize; 2], displs: [usize; 2]) -> Array2<f64> {
    let mut data: Array2<f64> = Array2::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            data[[i, j]] = ((i + displs[0]) + (j + displs[1]) * 10) as f64;
        }
    }
    data
}
