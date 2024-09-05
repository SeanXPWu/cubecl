use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{coop_id, lane_id, Dimensions, Offsets},
    block_io::{
        base::BlockWriter, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
    config::CmmaComptimeInfo,
};

#[cube]
pub(crate) fn write_to_output<F: Float>(
    out: &mut Tensor<F>,
    accumulators: Sequence<cmma::Matrix<F>>,
    offsets: Offsets,
    dims: Dimensions,
    config: Comptime<CmmaComptimeInfo>,
) {
    let accumulator_sm = fragment_to_shared_memory(accumulators, config);
    shared_memory_to_output(out, offsets, accumulator_sm, dims, config);
}

#[cube]
fn fragment_to_shared_memory<F: Float>(
    accumulators: Sequence<cmma::Matrix<F>>,
    config: Comptime<CmmaComptimeInfo>,
) -> SharedMemory<F> {
    let num_accumulators = Comptime::map(config, |c| c.num_accumulators);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let lane_dim = Comptime::map(config, |c| c.lane_dim);

    let sm_stride = tile_size * tile_size;
    let sm_size = num_accumulators * lane_dim * sm_stride;

    let mut acc_sm = SharedMemory::<F>::new(Comptime::get(sm_size));

    let coop_id = coop_id(config);
    let slice_offset = coop_id * Comptime::runtime(num_accumulators * sm_stride);

    for n in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
        let slice_start = slice_offset + n * Comptime::runtime(sm_stride);
        let slice_end = slice_start + Comptime::runtime(sm_stride);

        let slice = acc_sm.slice_mut(slice_start, slice_end);

        cmma::store::<F>(
            slice,
            accumulators.index(n),
            UInt::new(16),
            cmma::MatrixLayout::RowMajor,
        );
    }

    acc_sm
}

#[cube]
pub(crate) fn shared_memory_to_output<F: Float>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    config: Comptime<CmmaComptimeInfo>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_n_bounds) {
            write_tile::<F, WholeCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
        } else {
            write_tile::<F, VerticalCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
        }
    } else if Comptime::get(check_n_bounds) {
        write_tile::<F, HorizontalCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
    } else {
        write_tile::<F, UncheckedBlockIO>(out, offsets, accumulator_sm, dims, config);
    }
}

#[cube]
fn write_tile<F: Float, W: BlockWriter<F>>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    config: Comptime<CmmaComptimeInfo>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let tile_size_r = Comptime::runtime(tile_size);
    let num_accumulators = Comptime::map(config, |c| c.num_accumulators);
    let num_accumulators_r = Comptime::runtime(num_accumulators);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let num_accum_groups_in_block_row =
        Comptime::runtime(block_size_n / (tile_size * num_accumulators));

    let out_vec = Comptime::vectorization(out);
    let out_vec_r = Comptime::runtime(out_vec);
    let n_units_per_tile_row = Comptime::runtime(tile_size / out_vec);
    let sm_stride = Comptime::runtime(tile_size * tile_size);
    let coop_dim = Comptime::map(config, |c| c.coop_dim);

    let coop_id = coop_id(config);
    let lane_id = lane_id(config);

    let tile_row = coop_id / num_accum_groups_in_block_row;
    let tile_col = (coop_id % num_accum_groups_in_block_row) * num_accum_groups_in_block_row;

    let num_unit_writes = tile_size * tile_size / (out_vec * coop_dim);

    let read_offset = num_accumulators_r * coop_id * sm_stride + lane_id * out_vec_r;
    let sm_step = Comptime::runtime(coop_dim * out_vec);

    let lane_row_step = Comptime::runtime(coop_dim * out_vec / tile_size);
    let unit_write_row = lane_id / n_units_per_tile_row;
    let unit_write_col = lane_id % n_units_per_tile_row * out_vec_r;

    let row_offset = offsets.cube_row + tile_row * tile_size_r;
    let write_col = offsets.cube_col + tile_col * tile_size_r + unit_write_col;

    for i in range(0u32, Comptime::get(num_unit_writes), Comptime::new(true)) {
        let read_pos = read_offset + i * sm_step;
        let write_row = row_offset + unit_write_row + i * lane_row_step;

        for n in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
            W::write_output(
                out,
                accumulator_sm,
                n,
                offsets.batch_out,
                read_pos,
                write_row,
                write_col,
                dims,
                config,
            );
        }
    }
}
