use cubecl_core::prelude::*;

use super::config::MatmulConfig;

/// Provides configuration for a matmul kernel at any level
pub trait MatmulKernel<I: Numeric, O: Numeric> {
    /// Configuration tailored to the matmul implementation
    type Config: MatmulConfig;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: Self::Config);
}

/// Provides launch entry point to solve a matmul
pub trait MatmulLaunch<I: Numeric, O: Numeric>: MatmulKernel<I, O> {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: <Self as MatmulKernel<I, O>>::Config,
    );
}
