"""Issue #55: exercise the installed MTP config-copy helper and real aux guard.

Run inside the recipe image: python3 /tests/test_mtp_aux_output_cpu.py
No GPU or model weights. Extract the installed helper's AST to avoid importing
CUDA model classes; only its quantization dependencies are test doubles. A small
dataclass runs vLLM's actual aux-output compatibility guard on each replacement.
This checks configuration copying, not GPU boot or routed-expert delivery.
"""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import vllm
from vllm.config import VllmConfig
from vllm.config.aux_output import AuxOutputConfig


@dataclasses.dataclass
class ConfigFixture:
    model_config: object
    speculative_config: object
    aux_output_config: AuxOutputConfig
    cache_config: object = dataclasses.field(
        default_factory=lambda: NS(enable_prefix_caching=True)
    )
    parallel_config: object = dataclasses.field(
        default_factory=lambda: NS(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        )
    )
    use_v2_model_runner: bool = True
    kv_transfer_config: object = None
    quant_config: object = None

    def __post_init__(self):
        VllmConfig._verify_aux_output_compatibility(self)


def installed_helper():
    path = Path(vllm.__file__).parent / "models/qwen4_exp/nvidia/mtp.py"
    source = ast.parse(path.read_text())
    helper = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_make_draft_vllm_config"
    )
    namespace = {
        "VllmConfig": VllmConfig,
        "AuxOutputConfig": AuxOutputConfig,
        "replace": dataclasses.replace,
        "get_draft_quant_config": lambda config: None,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[helper.name]


class MtpAuxOutputTest(unittest.TestCase):
    def target(self, enabled=True):
        draft = NS(runner_type="draft", is_moe=True)
        return ConfigFixture(
            model_config=NS(runner_type="generate", is_moe=True),
            speculative_config=NS(
                draft_model_config=draft, enable_adaptive_verification=False
            ),
            aux_output_config=AuxOutputConfig(
                enable_return_routed_experts=enabled, max_bytes=4096
            ),
        )

    def test_real_guard_reproduces_inherited_aux_failure(self):
        target = self.target()
        with self.assertRaisesRegex(ValueError, "only supports generate runners"):
            dataclasses.replace(
                target, model_config=target.speculative_config.draft_model_config
            )

    def test_draft_disables_aux_without_mutating_target(self):
        target = self.target()
        original_aux = target.aux_output_config
        draft = installed_helper()(target, 48)
        self.assertIs(draft.model_config, target.speculative_config.draft_model_config)
        self.assertFalse(draft.aux_output_config.enabled)
        self.assertIsNone(draft.aux_output_config.max_bytes)
        self.assertIsNot(draft.aux_output_config, original_aux)
        self.assertIs(target.aux_output_config, original_aux)
        self.assertTrue(target.aux_output_config.enabled)
        self.assertEqual(target.aux_output_config.max_bytes, 4096)
        self.assertIs(draft.cache_config, target.cache_config)
        self.assertIs(draft.parallel_config, target.parallel_config)

    def test_aux_disabled_control(self):
        target = self.target(enabled=False)
        draft = installed_helper()(target, 48)
        self.assertFalse(draft.aux_output_config.enabled)
        self.assertFalse(target.aux_output_config.enabled)
        self.assertIs(draft.speculative_config, target.speculative_config)

    def test_target_restrictions_remain_enforced(self):
        target = self.target()
        with self.assertRaisesRegex(ValueError, "requires prefix caching"):
            dataclasses.replace(target, cache_config=NS(enable_prefix_caching=False))
        with self.assertRaisesRegex(ValueError, "adaptive speculative verification"):
            dataclasses.replace(
                target,
                speculative_config=NS(
                    draft_model_config=target.speculative_config.draft_model_config,
                    enable_adaptive_verification=True,
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
