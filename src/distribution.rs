//! # Data distribution
//!
//! Store size and first and last index of current processor and
//! of all participating processors along a single, possibly
//! split, dimension.
#![allow(clippy::similar_names)]

/// Distribute Grid points to processors.
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Size of data of current processor
    pub sz: usize,
    /// Starting index of data of current processor
    pub st: usize,
    /// Ending index of data of current processor
    pub en: usize,
    /// Size of data of all processors
    pub sz_procs: Vec<usize>,
    /// Starting index of data of all processors
    pub st_procs: Vec<usize>,
    /// Ending index of data of all processors
    pub en_procs: Vec<usize>,
}

impl Distribution {
    /// Generate new contiguous decomposition, i.e. sz = ``n_global``
    ///
    /// # Arguments
    /// * `n_global`: Total number of grid points [nx global, ny global]
    #[must_use]
    pub fn contiguous(n_global: usize) -> Self {
        // Get size and start/end index of current processor
        let st = 0;
        let en = n_global - 1;
        let sz = n_global;
        let st_procs: Vec<usize> = vec![st];
        let en_procs: Vec<usize> = vec![en];
        let sz_procs: Vec<usize> = vec![sz];

        Self {
            sz,
            st,
            en,
            sz_procs,
            st_procs,
            en_procs,
        }
    }

    /// Generate decomposition
    ///
    /// # Arguments
    /// * `n_global`: Total number of grid points [nx global, ny global]
    /// * `nprocs`: Number of processors
    /// * `nrank`: Current processor id
    #[must_use]
    pub fn split(n_global: usize, nprocs: usize, nrank: usize) -> Self {
        // Distribute
        let (st_procs, en_procs, sz_procs) = Self::distribute(n_global, nprocs);

        // Get size and start/end index of current processor
        let st = st_procs[nrank];
        let en = en_procs[nrank];
        let sz = sz_procs[nrank];

        Self {
            sz,
            st,
            en,
            sz_procs,
            st_procs,
            en_procs,
        }
    }

    /// Distribute grid points across processors along 1-dimension
    ///
    /// # Arguments
    /// * `n_global`: Total number of grid points along the split dimension
    /// * `nprocs`: Number of processors in the split dimension
    ///
    /// # Return
    /// Vectors containing starting/ending index and size of each
    /// processor
    fn distribute(n_global: usize, nprocs: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let size = n_global / nprocs;
        let mut st = vec![0; nprocs];
        let mut en = vec![0; nprocs];
        let mut sz = vec![0; nprocs];
        // Try to distribute N points
        st[0] = 0;
        sz[0] = size;
        en[0] = size - 1;
        // Distribute the rest if necessary
        let nu = n_global - size * nprocs;
        // Define how many processors held exactly N points, the rest holds N+1
        let nl = nprocs - nu;
        // Distribute N points on the first processors
        for i in 1..nl {
            st[i] = st[i - 1] + size;
            sz[i] = size;
            en[i] = en[i - 1] + size;
        }
        // Distribute  N+1 points on the last processors
        let size = size + 1;
        for i in nl..nprocs {
            st[i] = en[i - 1] + 1;
            sz[i] = size;
            en[i] = en[i - 1] + size;
        }
        // Very last processor
        en[nprocs - 1] = n_global - 1;
        sz[nprocs - 1] = en[nprocs - 1] - st[nprocs - 1] + 1;
        (st, en, sz)
    }
}
