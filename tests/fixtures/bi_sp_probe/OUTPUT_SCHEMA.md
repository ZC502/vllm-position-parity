# Output shape of `bi_sp_probe.py` (one JSON file per condition)

```jsonc
{
  "config": {                       // execution metadata
    "cond": "P0_tp2_sp1_triton",    // free-form condition name (argv[1])
    "tp": 2, "sp": "1",             // tensor_parallel_size; "off" or the explicit sp_min_token_num
    "backend": "TRITON_ATTN",       // attention backend
    "enable_sp": true, "sp_min_token_num": 1,   // as resolved by VllmConfig after validation (the #56377 gate flips these to false)
    "batch_invariant": "1",         // VLLM_BATCH_INVARIANT env as seen by the driver
    "prefix_caching": false,        // cache_config.enable_prefix_caching
    "cudagraph_mode": "CUDAGraphMode.FULL",
    "reverse": "0", "N": 64
  },
  "verdicts": [                     // four pairwise comparisons, all bitwise on float32 logprobs of the sampled token
    {
      "pair": "bs1_a_vs_bs1_b",     // also bsN_a_vs_bsN_b (repeat stability), bs1_a_vs_bsN_a, bs1_b_vs_bsN_b (batch composition)
      "tokens": 1536,               // N prompts x max_tokens compared
      "token_seq_mismatch_prompts": 0,   // prompts whose greedy token sequence differs
      "logprob_mismatch_tokens": 4,      // positions whose sampled-token logprob differs bitwise
      "max_abs_diff": 3.1e-06,
      "first_mismatches": [[34, 5, -0.123, -0.123001, 1e-06], ...]   // up to 12 x [prompt_idx, step, lp_a, lp_b, |diff|]
    }
  ],
  "runs": {                         // raw per-prompt data, prompt index preserved across runs
    "bs1_a": [[[tok0, tok1, ...], [lp0, lp1, ...]], ...],   // one [token_ids, logprobs] pair per prompt
    "bsN_a": ..., "bs1_b": ..., "bsN_b": ...
  }
}
```

Notes for a batch-composition axis: (1) keep prompt identity (index) across the single-request and batched executions; (2) compare the sampled token's logprob, not top-k sets, so greedy divergence is visible as `token_seq_mismatch_prompts`; (3) disable prefix caching for the batched-vs-single comparison unless caching is the variable under test (prompts that share a prefix change the number of newly computed tokens, which interacts with SP padding — see #56370); (4) record `enable_sp` / `sp_min_token_num` / `cudagraph_mode` from the resolved config rather than from the request, because validation may rewrite them.
