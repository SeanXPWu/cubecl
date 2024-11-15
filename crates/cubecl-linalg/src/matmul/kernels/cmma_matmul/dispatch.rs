use cubecl_core::{prelude::*, Feature};

use crate::matmul::components::stage::{S4x4x2, StageSize};
use crate::matmul::components::tile::accelerated::Accelerated16x16x16;
use crate::matmul::components::tile::plane::PlaneMma16x16x16;
use crate::matmul::components::tile::Matmul;
use crate::matmul::components::{tile, MatrixLayout};
use crate::matmul::components::{MatmulKernel, MatmulProblem};

/// Launch information for a matmul
pub trait MatmulLaunchDispatch {
    const PLANE_DIM: u32;
    type StageSize: StageSize;
    type ElementInput: Numeric;
    type ElementAccumulator: Numeric;

    type TileMatmul: tile::Matmul<Self::ElementInput, Self::ElementAccumulator>
        + MatmulKernel<Self::ElementInput, Self::ElementAccumulator>;

    fn cube_dim() -> CubeDim;
    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount;
    fn tile_config(
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> <Self::TileMatmul as MatmulKernel<Self::ElementInput, Self::ElementAccumulator>>::Config;
}

pub struct PlaneMmaLaunchDispatch {}

impl MatmulLaunchDispatch for PlaneMmaLaunchDispatch {
    const PLANE_DIM: u32 = 32;
    type StageSize = S4x4x2;
    type ElementInput = f32;
    type ElementAccumulator = f32;

    type TileMatmul = PlaneMma16x16x16<Self::ElementInput, Self::ElementAccumulator>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::K;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }

    fn tile_config(
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> tile::plane::Config {
        tile::plane::Config::new(
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        )
    }
}

pub struct CmmaLaunchDispatch {}

impl MatmulLaunchDispatch for CmmaLaunchDispatch {
    const PLANE_DIM: u32 = 32;
    type StageSize = S4x4x2;
    type ElementInput = half::f16;
    type ElementAccumulator = f32;

    type TileMatmul = Accelerated16x16x16<Self::ElementInput, Self::ElementAccumulator>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }

    fn tile_config(
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> tile::accelerated::Config {
        tile::accelerated::Config::new(
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        )
    }
}

/// Checks if the matmul cmma can be used.
pub fn check_availability<D: MatmulLaunchDispatch, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
) -> Result<(), ()> {
    if !client.properties().feature_enabled(Feature::Cmma {
        a: D::ElementInput::as_elem(),
        b: D::ElementInput::as_elem(),
        c: D::ElementAccumulator::as_elem(),
        m: D::TileMatmul::M as u8,
        k: D::TileMatmul::K as u8,
        n: D::TileMatmul::N as u8,
    }) {
        return Err(());
    }

    Ok(())
}
