"""vllm_modelopt_block_moe — FP8_BLOCK_SCALES layers in ModelOpt MIXED_PRECISION checkpoints.

NVIDIA's own Qwen3.8-Flash-Next-NVFP4 checkpoint quantizes the MTP drafter's routed experts
as blockwise fp8 (``quant_algo: FP8_BLOCK_SCALES, group_size: 128``: fp8 ``weight`` +
fp32 ``weight_scale_inv`` per 128x128 block, the DeepSeek-V3 layout) under a
``MIXED_PRECISION`` quantization config. vLLM's ``ModelOptMixedPrecisionConfig`` resolves
per-layer algorithms but only knows FP8 (per-tensor), FP8_PB_WO, NVFP4, W4A16_NVFP4 and
MXFP8; an FP8_BLOCK_SCALES layer falls through to "unquantized", the MoE layer is created
with bf16 parameters, and loading dies with
``Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'``.

vLLM already has the right kernels for this layout in its generic fp8 quantization
(``Fp8MoEMethod`` / ``Fp8LinearMethod`` with ``weight_block_size``), so this shim routes
those layers there. Two things are fixed:

  * algorithm: a layer whose resolved ``quant_algo`` is ``FP8_BLOCK_SCALES`` gets
    ``Fp8Config(weight_block_size=[g, g])`` with ``g`` = the entry's ``group_size``;
  * naming: vLLM instantiates the drafter as ``mtp.layers.<num_hidden_layers + i>`` while
    the checkpoint's ``quantized_layers`` says ``mtp.layers.<i>``; the exclude lists are
    already remapped by the model code, the quantized_layers dict is not. The lookup here
    tries the checkpoint-side name too.

Inert for checkpoints that declare no FP8_BLOCK_SCALES layer (RadixArk, Inferact, our
hybrid); the base class is untouched otherwise. ``VLLM_MODELOPT_BLOCK_MOE=0`` disables it.
"""
import logging
import os
import re

logger = logging.getLogger("vllm.modelopt_block_moe")
_SENTINEL = "_block_moe_patched"
_ALGO = "FP8_BLOCK_SCALES"
_MTP_RE = re.compile(r"^(?P<head>(?:.*\.)?mtp\.layers\.)(?P<idx>\d+)(?P<tail>(?:\..*)?)$")
# checkpoint-side keys for the drafter: mtp.layers.<i>... or, after the draft hf->vllm mapper, model.layers.<i>...
_KEY_RE = re.compile(r"^(?:model\.)?(?:mtp|model)\.layers\.(?P<idx>\d+)(?P<tail>(?:\..*)?)$")


def _enabled() -> bool:
    return os.environ.get("VLLM_MODELOPT_BLOCK_MOE", "1").lower() not in ("0", "false", "no")


def _num_hidden_layers() -> int | None:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().model_config
        text = getattr(cfg, "hf_text_config", None) or cfg.hf_config
        return int(getattr(text, "num_hidden_layers"))
    except Exception:  # pragma: no cover - outside an engine, or an unexpected config
        return None


def _checkpoint_names(prefix: str) -> list[str]:
    """Checkpoint-side spellings of a vLLM prefix.

    ``mtp.layers.<n>`` -> also ``mtp.layers.<n - L>`` (L = num_hidden_layers), and each of
    them with the ``mtp.`` head spelled ``model.`` as well: the draft model's hf->vllm mapper
    rewrites ``mtp.`` to ``model.`` and ``apply_vllm_mapper`` applies it to quantized_layers.
    """
    names = [prefix]
    if prefix.startswith("model.layers."):
        names.append("mtp." + prefix[len("model."):])
    L = _num_hidden_layers()
    for name in list(names):
        m = _MTP_RE.match(name)
        if m and L is not None and int(m.group("idx")) >= L:
            names.append(f"{m.group('head')}{int(m.group('idx')) - L}{m.group('tail')}")
    out = list(names)
    for name in names:
        if name.startswith("mtp."):
            out.append("model." + name[len("mtp."):])
    return list(dict.fromkeys(out))


_CKPT_LAYERS: dict | None = None


def _ckpt_quantized_layers() -> dict:
    """quantized_layers straight from the served checkpoint (hf_quant_config.json, else
    config.json's quantization_config), cached. Used by the RoutedExperts-level hook so the
    routing does not depend on which config object a model hands its expert layer."""
    global _CKPT_LAYERS
    if _CKPT_LAYERS is not None:
        return _CKPT_LAYERS
    _CKPT_LAYERS = {}
    try:
        import json
        from vllm.config import get_current_vllm_config

        path = get_current_vllm_config().model_config.model
        for fname in ("hf_quant_config.json", "config.json"):
            fp = os.path.join(path, fname)
            if not os.path.isfile(fp):
                continue
            d = json.load(open(fp))
            q = d.get("quantization", d.get("quantization_config", {})) or {}
            ql = q.get("quantized_layers") or {}
            if ql:
                _CKPT_LAYERS = dict(ql)
                break
    except Exception as exc:  # pragma: no cover
        logger.warning("modelopt block-moe: cannot read the checkpoint's quantized_layers: %s", exc)
    return _CKPT_LAYERS


def _lookup(quantized_layers: dict, prefix: str):
    """(quant_algo, entry) for a vLLM prefix against a quantized_layers dict (checkpoint naming)."""
    for name in _checkpoint_names(prefix):
        if name in quantized_layers:
            e = quantized_layers[name]
            return str(e.get("quant_algo", "")).upper(), e
        dot = name + "."
        for key, e in quantized_layers.items():
            if key.startswith(dot):
                return str(e.get("quant_algo", "")).upper(), e
    m = _MTP_RE.match(prefix) or (_MTP_RE.match("mtp." + prefix[len("model."):]) if prefix.startswith("model.layers.") else None)
    if m:
        tail = m.group("tail") or ""
        hits = [(int(km.group("idx")), e) for key, e in quantized_layers.items()
                if (km := _KEY_RE.match(key)) and (km.group("tail") or "") == tail]
        if len(hits) == 1:
            return str(hits[0][1].get("quant_algo", "")).upper(), hits[0][1]
        if len(hits) > 1:
            hits.sort()
            n = int(m.group("idx")); L = _num_hidden_layers()
            rank = (n - L) if L is not None and n >= L else 0
            e = hits[min(rank, len(hits) - 1)][1]
            return str(e.get("quant_algo", "")).upper(), e
    return None, None


def apply() -> None:
    if not _enabled():
        return
    from vllm.model_executor.layers.quantization import modelopt as m

    cls = m.ModelOptMixedPrecisionConfig
    if getattr(cls, _SENTINEL, False):
        return

    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod

    orig_resolve = cls._resolve_quant_algo
    orig_gqm = cls.get_quant_method

    def _entry_for(self, prefix: str):
        """(quant_algo, entry) for prefix.

        Order: the resolver's own candidates for every checkpoint-side spelling of the prefix;
        then, for an MTP prefix, the quantized_layers entry that has the same tail
        (``.mlp.experts``) under any ``mtp.layers.<i>`` / ``model.layers.<i>`` key — the
        checkpoint numbers the drafter from 0, vLLM from num_hidden_layers, and the exact
        offset is not always recoverable from inside the config object.
        """
        for name in _checkpoint_names(prefix):
            for cand in self._quantized_layer_prefix_candidates(name):
                if cand in self.quantized_layers:
                    e = self.quantized_layers[cand]
                    return str(e.get("quant_algo", "")).upper(), e
                dot = cand + "."
                for key, e in self.quantized_layers.items():
                    if key.startswith(dot):
                        return str(e.get("quant_algo", "")).upper(), e
        m = _MTP_RE.match(prefix) or _MTP_RE.match("mtp." + prefix[len("model."):]) if prefix.startswith(("mtp.", "model.layers.")) else None
        if m:
            tail = m.group("tail") or ""
            hits = []
            for key, e in self.quantized_layers.items():
                km = _KEY_RE.match(key)
                if km and (km.group("tail") or "") == tail:
                    hits.append((int(km.group("idx")), e))
            if len(hits) == 1:
                logger.info("modelopt block-moe: %s matched checkpoint entry by tail (%s)", prefix, hits[0][1])
                return str(hits[0][1].get("quant_algo", "")).upper(), hits[0][1]
            if len(hits) > 1:      # several MTP layers: pair by rank
                hits.sort()
                n = int(m.group("idx"))
                L = _num_hidden_layers()
                rank = (n - L) if L is not None and n >= L else 0
                e = hits[min(rank, len(hits) - 1)][1]
                return str(e.get("quant_algo", "")).upper(), e
        if "mtp." in prefix or prefix.startswith("model.layers."):
            logger.info("modelopt block-moe: no quantized_layers entry for %s (keys with 'layers.': %s)", prefix,
                        [k for k in self.quantized_layers if "mtp" in k or k.startswith("model.layers.")][:6])
        return None, None

    def _resolve_quant_algo(self, prefix: str):
        algo = orig_resolve(self, prefix)
        if algo is None:
            algo, _ = _entry_for(self, prefix)
        return algo

    def get_quant_method(self, layer, prefix):
        algo, entry = _entry_for(self, prefix)
        if ".experts" in prefix or os.environ.get("VLLM_MODELOPT_BLOCK_MOE_DEBUG"):
            logger.info("modelopt block-moe: get_quant_method(%s, %s) -> algo %s", type(layer).__name__, prefix, algo)
        if algo == _ALGO and not self.is_layer_excluded(prefix):
            g = int((entry or {}).get("group_size", 128) or 128)
            fp8 = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[g, g],
            )
            try:
                from vllm.model_executor.layers.fused_moe import RoutedExperts
            except Exception:  # pragma: no cover - older layout
                from vllm.model_executor.layers.fused_moe.layer import FusedMoE as RoutedExperts
            if isinstance(layer, RoutedExperts):
                logger.info("modelopt block-moe: %s -> Fp8MoEMethod (block %dx%d)", prefix, g, g)
                return Fp8MoEMethod(fp8, layer)
            if isinstance(layer, LinearBase):
                logger.info("modelopt block-moe: %s -> Fp8LinearMethod (block %dx%d)", prefix, g, g)
                return Fp8LinearMethod(fp8)
        return orig_gqm(self, layer, prefix)

    cls._resolve_quant_algo = _resolve_quant_algo
    cls.get_quant_method = get_quant_method
    setattr(cls, _SENTINEL, True)

    # Belt and braces: whatever config object (or None) a model hands its expert layer, an
    # expert layer the CHECKPOINT declares as FP8_BLOCK_SCALES gets the block-fp8 MoE method.
    try:
        from vllm.model_executor.layers.fused_moe import routed_experts as re_mod

        RE = re_mod.RoutedExperts
        if not getattr(RE, _SENTINEL, False):
            orig_re_gqm = RE._get_quant_method

            def _re_get_quant_method(self, prefix, quant_config, moe_config):
                algo, entry = _lookup(_ckpt_quantized_layers(), prefix)
                logger.info("modelopt block-moe: experts %s -> checkpoint algo %s (config %s)", prefix, algo, type(quant_config).__name__)
                if algo == _ALGO:
                    g = int((entry or {}).get("group_size", 128) or 128)
                    fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[g, g])
                    logger.info("modelopt block-moe: %s -> Fp8MoEMethod (block %dx%d) [RoutedExperts hook]", prefix, g, g)
                    return Fp8MoEMethod(fp8, self)
                return orig_re_gqm(self, prefix, quant_config, moe_config)

            RE._get_quant_method = _re_get_quant_method
            setattr(RE, _SENTINEL, True)
    except Exception as exc:  # pragma: no cover
        logger.warning("modelopt block-moe: RoutedExperts hook not installed: %s", exc)
    logger.info("modelopt block-moe: FP8_BLOCK_SCALES support wired into ModelOptMixedPrecisionConfig + RoutedExperts")
