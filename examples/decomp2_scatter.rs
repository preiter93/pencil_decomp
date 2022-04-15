//! cargo mpirun --np 2 --example decomp2_scatter
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

    // Scatter x
    let data = test_array(n_global, [0, 0]);
    let mut x_data = Array2::zeros(decomp2.x_pencil.shape());
    decomp2.scatter_x(&data, &mut x_data);
    assert_eq!(x_data, test_array_from_pencil(&decomp2.x_pencil));

    // Scatter y
    let data = test_array(n_global, [0, 0]);
    let mut y_data = Array2::zeros(decomp2.y_pencil.shape());
    decomp2.scatter_y(&data, &mut y_data);
    assert_eq!(y_data, test_array_from_pencil(&decomp2.y_pencil));
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
