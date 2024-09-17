use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::config::ComptimeCmmaInfo;
use super::cube_dispatch::base::{
    ColMajorCubeDispatch, CubeDispatch, RowMajorCubeDispatch, SwizzleCubeDispatch,
};

use crate::matmul::cmma::block_loop::{BlockLoop, DoubleBufferLoop, SingleBufferLoop};

#[cube(launch_unchecked)]
#[allow(unused_mut)]
pub fn cmma_kernel<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let ids = get_ids();
    let dims = get_dims::<F>(lhs, rhs);
    let offsets = calculate_offsets::<F>(lhs, rhs, out, comptime_info);
    let runtime_info = RuntimeCmmaInfo { ids, dims, offsets };

    if comptime_info.double_buffering {
        DoubleBufferLoop::block_loop::<F, FC>(lhs, rhs, out, runtime_info, comptime_info);
    } else {
        SingleBufferLoop::block_loop::<F, FC>(lhs, rhs, out, runtime_info, comptime_info);
    }
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Dimensions {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub coop: u32,
    pub lane: u32,
    pub team: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct RuntimeCmmaInfo {
    pub ids: Ids,
    pub dims: Dimensions,
    pub offsets: Offsets,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct SharedMemories<FC: Float> {
    pub lhs: SharedMemory<FC>,
    pub rhs: SharedMemory<FC>,
}

#[derive(CubeType, Copy, Clone)]
/// Not divided by vectorization factor
///
/// Note: batch offsets take stride into account, but not the others
pub(crate) struct Offsets {
    pub batch_lhs: u32,
    pub batch_rhs: u32,
    pub batch_out: u32,
    pub cube_row: u32,
    pub cube_col: u32,
}

#[derive(CubeType)]
pub(crate) struct CmmaMatrices<F: Float, FC: Float> {
    pub accumulators: Sequence<cmma::Matrix<F>>,
    pub lhs: cmma::Matrix<FC>,
    pub rhs: cmma::Matrix<FC>,
}

#[cube]
fn get_dims<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>) -> Dimensions {
    let rank = lhs.rank();
    let first_dim = rank - 2;
    let second_dim = rank - 1;
    let m = lhs.shape(first_dim);
    let k = lhs.shape(second_dim);
    let n = rhs.shape(second_dim);

    Dimensions { m, k, n }
}

#[cube]
fn calculate_offsets<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> Offsets {
    let (cube_row, cube_col) = get_row_col(comptime_info);

    let rank = out.rank();

    let dim_m = lhs.shape(rank - 2);
    let dim_n = rhs.shape(rank - 1);

    // Batch offset for output
    let batch_out = dim_m * dim_n * CUBE_POS_Z;
    let mut batch_lhs = 0;
    let mut batch_rhs = 0;

    // Batch offset for lhs, rhs
    for b in 0..rank - 2 {
        let tmp = batch_out / out.stride(b);
        batch_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        batch_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    Offsets {
        batch_lhs,
        batch_rhs,
        batch_out,
        cube_row,
        cube_col,
    }
}

#[cube]
pub(crate) fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
    if comptime_info.cube_dispatch == 0 {
        RowMajorCubeDispatch::get_row_col(comptime_info)
    } else if comptime_info.cube_dispatch == 1 {
        ColMajorCubeDispatch::get_row_col(comptime_info)
    } else {
        SwizzleCubeDispatch::get_row_col(comptime_info)
    }
}

#[cube]
pub(crate) fn make_shared_memories<FC: Float>(
    #[comptime] config: ComptimeCmmaInfo,
) -> SharedMemories<FC> {
    let block_size_m = config.block_size_m;
    let block_size_k = config.block_size_k;
    let block_size_n = config.block_size_n;

    let lhs = SharedMemory::<FC>::new(block_size_k * block_size_m);
    let rhs = SharedMemory::<FC>::new(block_size_k * block_size_n);

    SharedMemories::<FC> { lhs, rhs }
}

#[cube]
pub(crate) fn make_cmma_matrices<F: Float, FC: Float>(
    #[comptime] config: ComptimeCmmaInfo,
) -> CmmaMatrices<F, FC> {
    let num_accumulators = config.num_accumulators;
    let mut accumulators = Sequence::<cmma::Matrix<F>>::new();

    #[unroll]
    for _ in 0..num_accumulators {
        let acc = cmma::Matrix::<F>::new(
            cmma::MatrixIdent::Accumulator,
            16,
            16,
            16,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill::<F>(&acc, F::new(0.0));

        accumulators.push(acc);
    }

    let lhs = cmma::Matrix::<FC>::new(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
    );

    let rhs = cmma::Matrix::<FC>::new(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
    );

    CmmaMatrices::<F, FC> {
        accumulators,
        lhs,
        rhs,
    }
}

#[cube]
fn get_ids() -> Ids {
    Ids {
        coop: UNIT_POS_Y,
        lane: UNIT_POS_X,
        team: UNIT_POS_Y % 2,
    }
}
