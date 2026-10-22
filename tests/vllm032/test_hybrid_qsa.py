"""CPU regression: choose projection layout without importing GPU modules."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(__file__).resolve().parents[2] / 'src/vllm_fp8_hybrid_modelopt.py'
if source.exists():
    spec = importlib.util.spec_from_file_location('hybrid', source)
    hybrid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid)
else:
    import vllm_fp8_hybrid_modelopt as hybrid


class QSAProjectionLayout(unittest.TestCase):
    def config(self, layers):
        return SimpleNamespace(_is_fp8_layer=lambda name: name in layers)

    def test_fp8_qkv_keeps_checkpoint_layout(self):
        with patch.dict(os.environ, VLLM_FP8_HYBRID='1'):
            self.assertTrue(hybrid.qsa_requires_separate_projections(
                self.config({'layers.3.self_attn.qkv_proj'}), 'layers.3.self_attn'))

    def test_fp8_indexer_also_requires_separate_scales(self):
        with patch.dict(os.environ, VLLM_FP8_HYBRID='1'):
            self.assertTrue(hybrid.qsa_requires_separate_projections(
                self.config({'layers.3.self_attn.indexer.index_qk_proj'}), 'layers.3.self_attn'))

    def test_nonhybrid_layers_keep_upstream_fusion(self):
        with patch.dict(os.environ, VLLM_FP8_HYBRID='1'):
            for config in (None, SimpleNamespace(), self.config(set())):
                self.assertFalse(hybrid.qsa_requires_separate_projections(config, 'layers.3.self_attn'))

    def test_disabled_hybrid_never_changes_layout(self):
        with patch.dict(os.environ, VLLM_FP8_HYBRID='0'):
            self.assertFalse(hybrid.qsa_requires_separate_projections(
                self.config({'layers.3.self_attn.qkv_proj'}), 'layers.3.self_attn'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
