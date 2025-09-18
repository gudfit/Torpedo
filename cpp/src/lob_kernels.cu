#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "torpedocode/kernels.hpp"

namespace torpedocode {

namespace {

__global__ void fuse_features_kernel(const float *__restrict__ features,
                                     const float *__restrict__ topo_aligned,
                                     float *__restrict__ output,
                                     std::int64_t numel) {
  const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numel) return;
  output[idx] = features[idx] + topo_aligned[idx];
}

} // namespace

HybridForwardOutputs hybrid_forward_cuda(const HybridForwardInputs &inputs) {
  auto features = inputs.features.contiguous();
  auto topology = inputs.topology.contiguous();

  // Align topology to features: if last-dim differs, mean-reduce over last dim then expand
  if (features.sizes() != topology.sizes()) {
    if (topology.dim() >= 1) {
      auto reduced = topology.mean(-1, /*keepdim=*/true);
      topology = reduced.expand_as(features).contiguous();
    }
  }

  auto output = torch::empty_like(features);
  const auto numel = features.numel();
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  fuse_features_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      features.data_ptr<float>(), topology.data_ptr<float>(), output.data_ptr<float>(), numel);

  if (inputs.weights.defined() && inputs.weights.numel() == numel) {
    auto wv = inputs.weights.contiguous().view_as(output);
    output.mul_(wv);
  }

  return HybridForwardOutputs{output};
}

} // namespace torpedocode
