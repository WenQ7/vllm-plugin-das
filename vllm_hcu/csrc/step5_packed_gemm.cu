// SPDX-License-Identifier: Apache-2.0
#include "step5_packed_masked.cuh"
#include "step5_packed_contiguous.cuh"

void step5_fp8_packed_masked(torch::Tensor a, torch::Tensor weight,
    torch::Tensor a_scale, torch::Tensor weight_scale, torch::Tensor counts,
    torch::Tensor output) {
  vllm_hcu_step5_packed_masked::launch(
      a, weight, a_scale, weight_scale, counts, false, torch::Tensor(), output);
}

void step5_fp8_packed_contiguous(torch::Tensor a, torch::Tensor weight,
    torch::Tensor a_scale, torch::Tensor weight_scale, torch::Tensor indices,
    torch::Tensor output) {
  vllm_hcu_step5_packed_contiguous::run(
      a, weight, output, a_scale, weight_scale, indices, 256, 256, 4);
}
