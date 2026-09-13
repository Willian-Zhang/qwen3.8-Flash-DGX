#!/usr/bin/env python3
"""CPU unit test for the vllm#55513 backport added by patch_block_fp8_mtp.py (no GPU needed).

    docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx:v0.29 test_block_fp8_mtp_cpu.py

The PR's two unit tests, plus the v0.29 draft path end to end: an NVIDIA-style MIXED_PRECISION
config must resolve the draft's experts to block FP8, and a RadixArk-style NVFP4 config must come
out of the draft path exactly as before (MTP excluded, bf16 drafter).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.models.qwen4_exp.nvidia import mtp

DRAFT_EXPERTS = "mtp.layers.48.mlp.experts"  # runtime name of the checkpoint's mtp.layers.0.mlp.experts


def mixed(quantized_layers):
    return ModelOptMixedPrecisionConfig.from_config(
        {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": quantized_layers}}
    )


def draft_quant_config(quant_config):
    """Run the patched _make_draft_vllm_config on a stub VllmConfig (48 = num_hidden_layers)."""
    stub = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=object()))
    with patch.object(mtp, "get_draft_quant_config", return_value=quant_config), \
            patch.object(mtp, "replace", lambda cfg, **kw: SimpleNamespace(**kw)):
        return mtp._make_draft_vllm_config(stub, 48).quant_config


# 1. checkpoint MTP layer indices -> draft indices (vllm#55513, tests/models/qwen4_exp/test_config.py)
assert mtp._remap_quantized_layers(
    {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
     "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128}},
    48,
) == {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
      DRAFT_EXPERTS: {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128}}

# 2. both block-FP8 names dispatch to Fp8MoEMethod, [128, 128], dynamic (vllm#55513, test_modelopt.py)
for algo in ("FP8_PB_WO", "FP8_BLOCK_SCALES"):
    cfg = mixed({DRAFT_EXPERTS: {"quant_algo": algo, "group_size": 128}})
    layer, sentinel = MagicMock(spec=RoutedExperts), object()
    with patch.object(modelopt, "Fp8MoEMethod", return_value=sentinel) as method_cls:
        assert cfg.get_quant_method(layer, DRAFT_EXPERTS) is sentinel, algo
    fp8_cfg, called_layer = method_cls.call_args.args
    assert called_layer is layer
    assert fp8_cfg.weight_block_size == [128, 128] and fp8_cfg.activation_scheme == "dynamic"
try:
    mixed({"a.experts": {"quant_algo": "FP8_PB_WO", "group_size": 128},
           "b.experts": {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 64}})
except ValueError as exc:
    assert "one group_size" in str(exc), exc
else:
    raise AssertionError("block-FP8 layers with different group sizes were accepted")

# 3. the v0.29 draft path with an NVIDIA-style checkpoint (config.json quantization_config shape)
nvidia = ModelOptMixedPrecisionConfig.from_config({
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "ignore": ["lm_head", "model.language_model.layers.0.linear_attn*", "model.visual*"],
    "quantized_layers": {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "FP8"},
        "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO", "group_size": 128},
    },
})
assert nvidia._resolve_quant_algo(DRAFT_EXPERTS) is None  # what v0.29.0 sees: the key says layer 0
draft = draft_quant_config(nvidia)
assert draft._resolve_quant_algo(DRAFT_EXPERTS) == "FP8_PB_WO", draft.quantized_layers
with patch.object(modelopt, "Fp8MoEMethod", return_value="fp8-moe"):
    assert draft.get_quant_method(MagicMock(spec=RoutedExperts), DRAFT_EXPERTS) == "fp8-moe"

# 4. RadixArk-style NVFP4 checkpoint: no quantized_layers, the drafter stays excluded (bf16)
radix = ModelOptNvFp4Config.from_config({"quantization": {
    "quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
    "exclude_modules": ["lm_head", "mtp.*", "model.mtp.*"],
}})
draft = draft_quant_config(radix)
assert not hasattr(draft, "quantized_layers")
assert draft.get_quant_method(MagicMock(spec=RoutedExperts), DRAFT_EXPERTS) is None

print("block-FP8 MTP backport: OK (index remap, FP8_PB_WO/FP8_BLOCK_SCALES dispatch, "
      "NVIDIA-style draft path, NVFP4 draft path unchanged)")
