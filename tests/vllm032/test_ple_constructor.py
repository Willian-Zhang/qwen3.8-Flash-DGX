"""Exercise the installed mmap hook with v0.32's new constructor keyword."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch.nn as nn
import vllm_ple_mmap as mmap


class ConstructorContract(unittest.TestCase):
    def test_sequence_parallel_keyword_is_forwarded(self):
        module = ModuleType('qwen032_ple_fixture')
        module.Qwen4ExpPLEDeviceEmbedding = object()
        class Embedding(nn.Module):
            def __init__(self, config, embedding_dim, layer, max_tokens, *,
                         data_parallel_rank, prefix, quant_config, params_dtype,
                         use_sequence_parallel=False):
                super().__init__()
                self.use_sequence_parallel = use_sequence_parallel
                self.ngram_embedding = module.Qwen4ExpPLEDeviceEmbedding(32, embedding_dim)
                self.head_dim = embedding_dim
            def load_weights(self, weights):
                return set()
        Embedding.__module__ = module.__name__
        with patch.dict(sys.modules, {module.__name__:module}), patch.object(mmap, '_register_op'):
            mmap._install(Embedding)
            for enabled in (False, True):
                obj = Embedding(SimpleNamespace(), 16, 0, 8,
                                data_parallel_rank=0, prefix='layers.0.ple',
                                use_sequence_parallel=enabled)
                self.assertEqual(obj.use_sequence_parallel, enabled)
                self.assertEqual(obj.ngram_embedding.org_vocab_size, 32)
                self.assertIsNone(obj.ngram_embedding.table)
        mmap._REGISTRY.pop('layers.0.ple', None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
