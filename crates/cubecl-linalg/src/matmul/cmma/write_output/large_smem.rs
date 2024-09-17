use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::RuntimeCmmaInfo;

use super::{
    super::config::ComptimeCmmaInfo,
    base::{shared_memory_to_output, OutputWriter},
    smem_store::SmemStore,
};

pub(crate) struct LargeSmemWriter<S: SmemStore> {
    _s: PhantomData<S>,
}

#[cube]
impl<S: SmemStore> OutputWriter for LargeSmemWriter<S> {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let num_accumulators = comptime_info.num_accumulators;
        let tile_size = comptime_info.tile_size;
        let num_coops = comptime_info.num_coops;
        let ids = runtime_info.ids;

        let smem_stride = tile_size * tile_size;
        let smem_size = num_accumulators * num_coops * smem_stride;

        let mut acc_sm = SharedMemory::<F>::new(smem_size);

        let slice_offset = ids.coop * num_accumulators * smem_stride;
        let smem_position_base = num_accumulators * ids.coop;

        #[unroll]
        for n in 0..num_accumulators {
            let slice_start = slice_offset + n * smem_stride;
            let slice_end = slice_start + smem_stride;

            let slice = acc_sm.slice_mut(slice_start, slice_end);

            S::store(slice, accumulators.index(n), runtime_info.ids);
        }

        #[unroll]
        for n in 0..num_accumulators {
            let smem_position = smem_position_base + n;
            shared_memory_to_output(out, smem_position, acc_sm, n, runtime_info, comptime_info);
        }
    }
}
