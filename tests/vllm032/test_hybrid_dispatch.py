"""Installed ModelOpt hybrid dispatch against v0.32's four-member QKV map."""
import os
import unittest
from unittest.mock import patch

import vllm_fp8_hybrid_modelopt as hybrid
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config


class HybridDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(ModelOptNvFp4Config, hybrid._SENTINEL, False):
            hybrid._patch(ModelOptNvFp4Config)

    def test_bf16_indexer_does_not_hide_fp8_qkv(self):
        config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        config.packed_modules_mapping = {'qkv_proj':['q_proj','k_proj','v_proj','indexer.index_qk_proj']}
        config.fp8_layers = {f'model.layers.3.self_attn.{name}' for name in ('q_proj','k_proj','v_proj')}
        prefix = 'language_model.model.layers.3.self_attn'
        with patch.dict(os.environ, VLLM_FP8_HYBRID='1'):
            self.assertFalse(config._is_fp8_layer(prefix+'.qkv_proj'))
            proxy = hybrid.excluded_quant_config(config)
            self.assertIsNotNone(proxy)
            self.assertTrue(config._is_fp8_layer(prefix+'.qkv_proj'))
            self.assertTrue(hybrid.qsa_requires_separate_projections(config,prefix))
            self.assertFalse(config._is_fp8_layer(prefix+'.indexer.index_qk_proj'))
            self.assertEqual(config.packed_modules_mapping['qkv_proj'],['q_proj','k_proj','v_proj'])

    def test_disabled_hybrid_preserves_upstream_mapping(self):
        config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        mapping = ['q_proj','k_proj','v_proj','indexer.index_qk_proj']
        config.packed_modules_mapping = {'qkv_proj':mapping.copy()}
        with patch.dict(os.environ, VLLM_FP8_HYBRID='0'):
            self.assertIsNone(hybrid.excluded_quant_config(config))
            self.assertEqual(config.packed_modules_mapping['qkv_proj'],mapping)


if __name__ == '__main__':
    unittest.main(verbosity=2)
