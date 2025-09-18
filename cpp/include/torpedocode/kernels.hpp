#pragma once

#include <torch/extension.h>

namespace torpedocode {

struct HybridForwardInputs {
  torch::Tensor features;
  torch::Tensor topology;
  torch::Tensor weights;
};

struct HybridForwardOutputs {
  torch::Tensor fused;
};

HybridForwardOutputs hybrid_forward_cpu(const HybridForwardInputs &inputs);
HybridForwardOutputs hybrid_forward_cuda(const HybridForwardInputs &inputs);
HybridForwardOutputs hybrid_forward(const HybridForwardInputs &inputs);

} // namespace torpedocode
