use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{CmmaMatrices, RuntimeCmmaInfo, SharedMemories},
    compute_loop::compute_loop,
    config::ComptimeCmmaInfo,
    load_shared_memory::load_to_shared_memories,
    write_output::{base::OutputWriter, large_smem::LargeSmemWriter, reuse_smem::ReuseSmemWriter},
};

#[cube]
pub(crate) trait BlockLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        shared_memories: SharedMemories<FC>,
        cmma_matrices: CmmaMatrices<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
}

pub(crate) struct SingleBufferLoop {}
pub(crate) struct DoubleBufferLoop {}

#[cube]
impl BlockLoop for SingleBufferLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        shared_memories: SharedMemories<FC>,
        mut cmma_matrices: CmmaMatrices<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let block_size_k = comptime_info.block_size_k;
        let write_out_reuse_smem = comptime_info.write_out_reuse_smem;

        // Equals ceil(dims.k / block_size_k)
        let dims = runtime_info.dims;
        let num_loops = (dims.k + block_size_k - 1) / block_size_k;

        for block in 0..num_loops {
            let k_offset = block * block_size_k;

            load_to_shared_memories::<F, FC>(
                lhs,
                rhs,
                k_offset,
                shared_memories,
                runtime_info,
                comptime_info,
            );

            sync_units();

            compute_loop::<F, FC>(
                shared_memories,
                &mut cmma_matrices,
                runtime_info.ids,
                comptime_info,
            );

            sync_units();
        }

        if write_out_reuse_smem {
            ReuseSmemWriter::write_to_output(
                out,
                cmma_matrices.accumulators,
                runtime_info,
                comptime_info,
            );
        } else {
            LargeSmemWriter::write_to_output(
                out,
                cmma_matrices.accumulators,
                runtime_info,
                comptime_info,
            );
        }
    }
}

#[cube]
impl BlockLoop for DoubleBufferLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        shared_memories: SharedMemories<FC>,
        mut cmma_matrices: CmmaMatrices<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        // let block_size_k = comptime_info.block_size_k;
        // let write_out_reuse_smem = comptime_info.write_out_reuse_smem;

        // // Equals ceil(dims.k / block_size_k)
        // let dims = runtime_info.dims;
        // let num_loops = (dims.k + block_size_k - 1) / block_size_k;

        // for block in 0..num_loops {
        //     let k_offset = block * block_size_k;

        //     load_to_shared_memories::<F, FC>(
        //         lhs,
        //         rhs,
        //         k_offset,
        //         shared_memories,
        //         runtime_info,
        //         comptime_info,
        //     );

        //     sync_units();

        //     compute_loop::<F, FC>(
        //         shared_memories,
        //         &mut cmma_matrices,
        //         runtime_info.ids,
        //         comptime_info,
        //     );

        //     sync_units();
        // }

        // if write_out_reuse_smem {
        //     ReuseSmemWriter::write_to_output(
        //         out,
        //         cmma_matrices.accumulators,
        //         runtime_info,
        //         comptime_info,
        //     );
        // } else {
        //     LargeSmemWriter::write_to_output(
        //         out,
        //         cmma_matrices.accumulators,
        //         runtime_info,
        //         comptime_info,
        //     );
        // }
    }
}
