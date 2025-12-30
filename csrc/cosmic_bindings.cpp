#include <torch/extension.h>

#include "cosmic_fused.hpp"
#include "quant4.hpp"
#include "signals.hpp"

#include <tuple>
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cosmic CPU-only C++ extension";

    // Fused COSMIC dual-EMA step (main optimized path)
    m.def(
        "cosmic_fused_step",
        &cosmic::cosmic_fused_step,
        "Fused COSMIC step with dual-EMA momentum and adaptive LR",
        py::arg("params"),
        py::arg("grads"),
        py::arg("ema_shorts"),
        py::arg("ema_longs"),
        py::arg("exp_avg_sqs"),
        py::arg("blends"),
        py::arg("effective_lrs"),
        py::arg("step"),
        py::arg("decay_short"),
        py::arg("decay_long"),
        py::arg("beta2"),
        py::arg("eps"),
        py::arg("weight_decay"));

    m.def(
        "cosmic_fused_step_quant4",
        &cosmic::cosmic_fused_step_quant4,
        "Fused COSMIC step with quantized EMA state",
        py::arg("params"),
        py::arg("grads"),
        py::arg("ema_short_qs"),
        py::arg("ema_short_scales"),
        py::arg("ema_long_qs"),
        py::arg("ema_long_scales"),
        py::arg("exp_avg_sqs"),
        py::arg("blends"),
        py::arg("effective_lrs"),
        py::arg("step"),
        py::arg("decay_short"),
        py::arg("decay_long"),
        py::arg("beta2"),
        py::arg("eps"),
        py::arg("weight_decay"),
        py::arg("block_size"));

    m.def(
        "cosmic_fused_step_sign",
        &cosmic::cosmic_fused_step_sign,
        "Fused sign-gradient step (no momentum state)",
        py::arg("params"),
        py::arg("grads"),
        py::arg("effective_lrs"),
        py::arg("weight_decay"));

    // Expose quantization helpers.
    m.def("quant4_pack", &cosmic::quant4_pack, "Pack a float tensor into 4-bit values");
    m.def("quant4_unpack", &cosmic::quant4_unpack, "Unpack 4-bit values into float32");
    m.def(
        "quant4_pack_many",
        &cosmic::quant4_pack_many,
        "Pack a list of float tensors into 4-bit values");
    m.def(
        "quant4_unpack_many",
        &cosmic::quant4_unpack_many,
        "Unpack a list of 4-bit tensors into float32 tensors");
    m.def(
        "quant4_pack_blocks",
        [](const torch::Tensor& input, int64_t block_size, bool use_zero_point) {
            torch::Tensor packed;
            cosmic::Quant4Metadata meta;
            std::tie(packed, meta) = cosmic::quant4_pack_blocks(input, block_size, use_zero_point);
            return std::make_tuple(packed, meta.scales, meta.numel, meta.block_size, meta.use_zero_point);
        },
        "Pack a float tensor into 4-bit values with per-block scales",
        py::arg("input"),
        py::arg("block_size"),
        py::arg("use_zero_point") = false);
    m.def(
        "quant4_unpack_blocks",
        [](const torch::Tensor& packed,
           const torch::Tensor& scales,
           int64_t numel,
           int64_t block_size,
           bool use_zero_point) {
            cosmic::Quant4Metadata meta{scales, torch::zeros({scales.numel()}, torch::kInt16), numel, block_size, use_zero_point};
            return cosmic::quant4_unpack_blocks(packed, meta);
        },
        "Unpack 4-bit values with per-block scales",
        py::arg("packed"),
        py::arg("scales"),
        py::arg("numel"),
        py::arg("block_size"),
        py::arg("use_zero_point") = false);

    // Expose a basic tier signal for reassignment heuristics.
    m.def("tier_signal", &cosmic::tier_signal, "Compute a simple tier signal");
}
