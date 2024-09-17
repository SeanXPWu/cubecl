use cubecl_core::prelude::*;

// It is assumed that CMMA uses 32 units to compute 16x16x16 tiles
// TODO put it in config and split tile size into three different parameters
// TODO add number of smem banks
pub(crate) const CMMA_COOP_DIM: usize = 32;
pub(crate) const CMMA_TILE_SIZE: usize = 16;

#[derive(PartialEq, Eq, Clone, Copy)]
/// Defines how data travels from accumulators to global output
pub enum WriteOutStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    LargeSmem,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    ReuseSmem,
}

impl From<WriteOutStrategy> for bool {
    fn from(value: WriteOutStrategy) -> Self {
        match value {
            WriteOutStrategy::LargeSmem => false,
            WriteOutStrategy::ReuseSmem => true,
        }
    }
}

/// How cubes are dispatched in the hypercube
/// Should impact L2 cache reuse
#[derive(Clone, Copy)]
pub enum CubeDispatchStrategy {
    /// Cubes are dispatched row major
    RowMajor,
    /// Cubes are dispatched col major
    ColMajor,
    /// Cubes follow swizzle pattern, see https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85
    Swizzle,
}

impl From<CubeDispatchStrategy> for u32 {
    fn from(value: CubeDispatchStrategy) -> Self {
        match value {
            CubeDispatchStrategy::RowMajor => 0,
            CubeDispatchStrategy::ColMajor => 1,
            CubeDispatchStrategy::Swizzle => 2,
        }
    }
}

/// Defines how many shared memories are loaded/computed in parallel
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum BufferingStrategy {
    /// No parallelism
    Single,
    /// Computation on buffer 1 can occur during loading of buffer 2, and vice versa
    ///
    /// Eventual sub-configs:
    /// - Warps that work together (team) either consecutive (better coalescing?)
    /// or interleaved (better load balancing?)
    /// - The two teams either share the shared memory with an offset, or have their own
    /// (simpler, but maybe heavier)
    /// - Accumulation is performed in accumulators then two are summed only at the end
    /// when writing to SMEM, or the SMEM is written to all along (probably worse)
    Double,
}

impl From<BufferingStrategy> for bool {
    fn from(value: BufferingStrategy) -> Self {
        match value {
            BufferingStrategy::Single => false,
            BufferingStrategy::Double => true,
        }
    }
}

pub struct CmmaConfig {
    /// Corresponds to the number of tiles in the m and n dimensions for a block
    pub b_mn: usize,
    /// Corresponds to the number of tiles in the k dimension for a block
    pub b_k: usize,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
    /// Whether to write all accumulators in different spots of a large shared memory or reuse the space
    pub write_out: WriteOutStrategy,
    /// Order in which to dispatch cubes
    pub cube_dispatch: CubeDispatchStrategy,
    /// Whether to use double buffering or not
    pub buffering: BufferingStrategy,
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self::new(
            128,
            16,
            false,
            WriteOutStrategy::LargeSmem,
            CubeDispatchStrategy::ColMajor,
            BufferingStrategy::Double,
        )
    }
}

impl CmmaConfig {
    pub(crate) fn new(
        b_mn: usize,
        b_k: usize,
        unroll: bool,
        write_out: WriteOutStrategy,
        cube_dispatch: CubeDispatchStrategy,
        buffering: BufferingStrategy,
    ) -> CmmaConfig {
        assert!(b_mn % CMMA_TILE_SIZE == 0);
        assert!(b_k % CMMA_TILE_SIZE == 0);
        assert!(b_mn % b_k == 0);

        assert!(
            !(buffering == BufferingStrategy::Double && write_out == WriteOutStrategy::ReuseSmem)
        );

        CmmaConfig {
            b_mn,
            b_k,
            unroll,
            write_out,
            cube_dispatch,
            buffering,
        }
    }

    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> ComptimeCmmaInfo {
        let num_coops = self.b_mn * self.b_k / (CMMA_TILE_SIZE * CMMA_TILE_SIZE);
        ComptimeCmmaInfo {
            block_size_m: self.b_mn as u32,
            block_size_k: self.b_k as u32,
            block_size_n: self.b_mn as u32,
            tile_size: CMMA_TILE_SIZE as u32,
            unroll: self.unroll,
            check_m_bounds: m % self.b_mn != 0,
            check_k_bounds: k % self.b_k != 0,
            check_n_bounds: n % self.b_mn != 0,
            coop_dim: CMMA_COOP_DIM as u32,
            num_coops: num_coops as u32,
            num_accumulators: (self.b_mn / self.b_k) as u32,
            write_out_reuse_smem: self.write_out.into(),
            cube_dispatch: self.cube_dispatch.into(),
            double_buffering: self.buffering.into(),
        }
    }

    pub(crate) fn cube_count<R: Runtime>(
        &self,
        output_shape: &[usize],
    ) -> CubeCount<<R as Runtime>::Server> {
        let rank = output_shape.len();
        let num_rows = *output_shape.get(rank - 2).unwrap();
        let num_cols = *output_shape.get(rank - 1).unwrap();

        let cubes_x = f32::ceil(num_rows as f32 / self.b_mn as f32) as u32;
        let cubes_y = f32::ceil(num_cols as f32 / self.b_mn as f32) as u32;

        let mut num_iter = 1;
        for shape in output_shape.iter().take(rank - 2) {
            num_iter *= shape;
        }

        CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
    }

    pub(crate) fn cube_dim(&self) -> CubeDim {
        CubeDim {
            x: CMMA_COOP_DIM as u32,
            y: ((self.b_mn * self.b_k) / (CMMA_TILE_SIZE * CMMA_TILE_SIZE)) as u32,
            z: 1,
        }
    }

    pub(crate) fn available_vectorizations(&self) -> Vec<u8> {
        let vectorizations = vec![8, 4, 2];
        for v in vectorizations.iter() {
            assert!(CMMA_TILE_SIZE * CMMA_TILE_SIZE % (*v as usize * CMMA_COOP_DIM) == 0);
        }
        vectorizations
    }
}

impl Init for ComptimeCmmaInfo {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ComptimeCmmaInfo {
    /// Block size along dimension of lhs
    pub block_size_m: u32,
    /// Block size along common dimension
    pub block_size_k: u32,
    /// Block size along dimension of rhs
    pub block_size_n: u32,
    /// Tile size (dimension of one side). Should correspond to cmma supported tile size
    pub tile_size: u32,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Unroll
    pub unroll: bool,
    /// The number of units that can collaborate
    pub coop_dim: u32,
    /// The number of collaboration groups
    pub num_coops: u32,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: u32,
    /// Write out strategy: false = large, true = reuse
    pub write_out_reuse_smem: bool,
    /// 0 = RowMajor, 1 = ColMajor, 2 = Swizzle
    pub cube_dispatch: u32,
    /// Whether to use double buffering
    pub double_buffering: bool,
}
