#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "torpedocode/kernels.hpp"

namespace torpedocode {
namespace {

__global__ void zero_embeddings_kernel(float *embeddings, std::int64_t numel) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  embeddings[idx] = 0.0f;
}

} // namespace

RollingTopoOutputs rolling_topo_cuda(const RollingTopoInputs &inputs) {
  TORCH_CHECK(inputs.series.defined(), "series tensor must be defined");
  TORCH_CHECK(inputs.series.is_cuda(),
              "rolling_topo_cuda expects tensors on CUDA device");
  TORCH_CHECK(inputs.embedding_dim >= 0,
              "embedding_dim must be non-negative");

  auto series = inputs.series.contiguous();
  if (series.dtype() != torch::kFloat32) {
    series = series.to(torch::kFloat32);
  }
  const auto device = series.device();
  const auto num_rows = series.size(0);
  auto options = series.options().dtype(torch::kFloat32).device(device);
  auto embeddings = torch::empty({num_rows, inputs.embedding_dim}, options);

  const auto numel = embeddings.numel();
  if (numel > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    zero_embeddings_kernel<<<blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
        embeddings.data_ptr<float>(), numel);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "zero_embeddings_kernel launch failed");
  }

  return RollingTopoOutputs{std::move(embeddings)};
}

} // namespace torpedocode
