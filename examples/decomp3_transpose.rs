//! cargo mpirun --np 6 --example decomp3_transpose
use mpi::topology::Communicator;
use ndarray::Array3;
use pencil_decomp::{Decomp3, Pencil};

fn main() {
    // Init Mpi
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    assert!(world.size() == 6, "Run with 6 processors");

    // Parameters
    let n_global = [6, 7, 9];
    let cart_dims = [2, 3];
    let cart_periodic = [false, false];

    // Decomp
    let decomp3 = Decomp3::new(&universe, n_global, cart_dims, cart_periodic);

    // Test arrays
    let mut x_data: Array3<f64> = test_array_from_pencil(&decomp3.x_pencil);
    let mut y_data: Array3<f64> = Array3::zeros(decomp3.y_pencil.shape());
    let mut z_data: Array3<f64> = Array3::zeros(decomp3.z_pencil.shape());

    // Transpose x -> y
    decomp3.transpose_x_to_y(&x_data, &mut y_data);
    assert_eq!(y_data, test_array_from_pencil(&decomp3.y_pencil));

    // Transpose y -> z
    decomp3.transpose_y_to_z(&y_data, &mut z_data);
    assert_eq!(z_data, test_array_from_pencil(&decomp3.z_pencil));

    // Transpose z -> y
    y_data.fill(0.);
    decomp3.transpose_z_to_y(&z_data, &mut y_data);
    assert_eq!(y_data, test_array_from_pencil(&decomp3.y_pencil));

    // Transpose y -> x
    x_data.fill(0.);
    decomp3.transpose_y_to_x(&y_data, &mut x_data);
    assert_eq!(x_data, test_array_from_pencil(&decomp3.x_pencil));
}

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
