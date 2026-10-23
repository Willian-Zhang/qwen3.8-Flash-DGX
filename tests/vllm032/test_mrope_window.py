"""Installed M-RoPE cache bounds and exact numerical preservation on CPU."""
from contextlib import contextmanager
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope, _ROPE_DICT
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)

torch.set_num_threads(2)


@contextmanager
def context(served, *, draft=False, own_window=262144):
    target = NS(max_model_len=served)
    config = NS(
        compilation_config=CompilationConfig(custom_ops=["none"]),
        model_config=NS(max_model_len=own_window) if draft else target,
        speculative_config=NS(target_model_config=target) if draft else None,
    )
    with set_current_vllm_config(config), torch.device("cpu"):
        yield


def make(base=1024, scale=None, factor=1, dtype=torch.float32):
    return MRotaryEmbedding(
        head_size=32, rotary_dim=8, max_position_embeddings=base,
        base=10000000, is_neox_style=True, dtype=dtype,
        positions_factor=factor, scaling_factor=scale,
    )


def original_cache(rope):
    if rope.scaling_factor is None:
        return RotaryEmbeddingBase._compute_cos_sin_cache(rope)
    return YaRNScalingRotaryEmbedding._compute_cos_sin_cache(rope)


class MRoPEWindow(unittest.TestCase):
    def tearDown(self):
        _ROPE_DICT.clear()

    def test_real_native_boundary_and_last_served_position_are_addressable(self):
        with context(500000):
            rope = make(base=262144)
            self.assertGreaterEqual(rope.cos_sin_cache.shape[0], 500000)
            rows = rope.cos_sin_cache[[262143, 262144, 336000, 499999]]
            self.assertTrue(torch.isfinite(rows).all())
            self.assertEqual(rope.max_position_embeddings, 262144)

    def test_drafter_uses_target_window_instead_of_own_native_limit(self):
        with context(500000, draft=True, own_window=262144):
            rope = make(base=262144)
            self.assertGreaterEqual(rope.cos_sin_cache.shape[0], 500000)

    def test_native_window_is_bit_identical(self):
        with context(1024):
            rope = make()
            self.assertEqual(rope.cos_sin_cache.shape[0], 1024)
            self.assertTrue(torch.equal(rope.cos_sin_cache, original_cache(rope)))

    def test_growing_yarn_cache_preserves_base_frequencies_and_existing_rows(self):
        with context(10000):
            rope = make(scale=2.0)
            self.assertEqual(rope.max_position_embeddings, 1024)
            reference = original_cache(rope)
            self.assertEqual(reference.shape[0], 2048)
            self.assertEqual(rope.cos_sin_cache.shape[0], 10000)
            self.assertTrue(torch.equal(rope.cos_sin_cache[:2048], reference))
            inv = YaRNScalingRotaryEmbedding._compute_inv_freq(rope, 2.0)
            frequencies = torch.einsum("i,j -> ij", torch.tensor([9999.]), inv)
            expected = torch.cat((frequencies.cos() * rope.mscale,
                                  frequencies.sin() * rope.mscale), dim=-1)
            self.assertTrue(torch.equal(rope.cos_sin_cache[9999:], expected))

    def test_already_scaled_cache_is_unchanged(self):
        with context(2000):
            rope = make(scale=4.0)
            self.assertEqual(rope.cos_sin_cache.shape[0], 4096)
            self.assertEqual(rope.max_position_embeddings, 1024)
            self.assertTrue(torch.equal(rope.cos_sin_cache, original_cache(rope)))

    def test_multimodal_headroom_is_not_shrunk(self):
        with context(2000):
            rope = make(factor=4)
            self.assertEqual(rope.cos_sin_cache.shape[0], 4096)
            self.assertTrue(torch.equal(rope.cos_sin_cache, original_cache(rope)))

    def test_no_engine_context_preserves_direct_constructor_behavior(self):
        with context(10000), patch(
            "vllm.config.get_current_vllm_config_or_none", return_value=None
        ):
            rope = make()
            self.assertEqual(rope.cos_sin_cache.shape[0], 1024)

    def test_factory_covers_yarn_target_and_default_mtp_drafter(self):
        params = {"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                  "mrope_section": [1, 1, 2], "mrope_interleaved": True}
        with context(500000):
            target = get_rope(32, 262144, rope_parameters={
                **params, "rope_type": "yarn", "factor": 4.0,
                "original_max_position_embeddings": 262144,
            }, dtype=torch.float32, mrope_positions_factor=1)
        with context(500000, draft=True):
            draft = get_rope(32, 262144, rope_parameters={
                **params, "rope_type": "default",
            }, dtype=torch.float32, mrope_positions_factor=1)
        self.assertGreaterEqual(target.cos_sin_cache.shape[0], 500000)
        self.assertGreaterEqual(draft.cos_sin_cache.shape[0], 500000)
        self.assertEqual(target.max_position_embeddings, 262144)
        self.assertEqual(draft.max_position_embeddings, 262144)

    def test_bfloat16_cache_keeps_exact_old_rows(self):
        with context(4096):
            rope = make(dtype=torch.bfloat16)
            self.assertEqual(rope.cos_sin_cache.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(rope.cos_sin_cache[:1024],
                                        original_cache(rope).to(torch.bfloat16)))

    def test_served_window_is_per_constructor(self):
        with context(4096):
            extended = make()
        with context(1024):
            native = make()
        self.assertEqual(extended.cos_sin_cache.shape[0], 4096)
        self.assertEqual(native.cos_sin_cache.shape[0], 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
