# Fixture provenance

These fixtures were provided by @LioEinaudi in vllm-project/vllm#56370
for validating the batch-composition adapter.

They were captured from the same job using:

- vLLM v0.29.0 wheel without the #56377 gate
- 4x RTX PRO 6000 Blackwell (sm_120)
- Qwen/Qwen3-1.7B
- VLLM_BATCH_INVARIANT=1

The adapter treats the recorded `verdicts` only as a validation oracle.
All measurement summaries are independently recomputed from `runs.*`.
