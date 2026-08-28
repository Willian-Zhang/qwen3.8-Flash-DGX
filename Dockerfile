# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and appends one patch:
# it serves the 51B-parameter n-gram ("PLE") table from disk via mmap instead of
# keeping it resident in the 128 GB unified pool. That is the single change that
# lets the ~176B (122 GiB NVFP4) checkpoint fit next to a real KV cache on one box.
#
#   docker build -t qwen38-flash-dgx .
#
# The base image is multi-arch (arm64 for the Spark's Grace CPU). Pinned by digest
# for reproducibility; bump the tag below if the upstream recipe moves.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130,
# numpy 2.2.6 — the patch needs numpy, already present).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py

# Append the hook to the model file. No-op unless VLLM_PLE_MMAP=1 at runtime, so
# the image still behaves exactly like upstream when the flag is off.
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# --- GB10 FLA fixes, contributed by @Saren-Arterius
#     (https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) ---
# 1) sm_121 reports 99 KiB of shared memory per block; the flash-linear-attention
#    gate asks for 100 KiB, so all 36 GDN layers silently fell back to small tiles.
#    Lowering the gate to 99 KiB lets the GB10 take the big-tile path.
# 2) fla#953: tl.dot race on Blackwell with num_warps=4 in chunk_delta_h -> pin 2.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched" \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"
