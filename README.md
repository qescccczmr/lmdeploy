# PR 4968 opt_plan2 validation and Perfetto captures

Actual browser screenshots of https://ui.perfetto.dev/; rank0, trial0, 2K/8K, both engines MTP5. LMDeploy ec2ec448, vLLM 606d124b. CPU+CUDA profiling is separate from the 30-request unprofiled timing. Client timestamps are added after capture from real clock anchors. Selected Duration/4 gives client TPOT; GPU graph spans are first-to-last correlated kernel intervals and are not TPOT.

No opt_plan2 candidate was retained in production: metadata fusion has no significant serving gain; split-K native-head attention is faster in isolation but changes generated tokens and fixed-history logits. Existing accepted optimizations are preserved. Candidate source changes are not in the PR.

Perfetto overlapping-complete-event warnings are preserved. Source hashes, SQL checks, client windows and graph kernel counts are in capture_manifest.json. The selected intervals and displayed verification graph are verified against the original JSON.

# opt_plan2: assessment and measured disposition

No new candidate is enabled in production; the final worktree is clean at ec2ec448. Existing accepted optimizations remain combined. See assessment.md for review, metadata_ab for paired timing and exact logits, native_validation.json for operator checks, and sparse_teacherforce/summary.json for aligned full-model errors. Candidate diffs are retained only as external report artifacts.

The cumulative-metadata prototype has no significant end-to-end gain. The native-head split-K prototype is faster at the operator level but changes unconstrained 2K generated tokens and fails fixed-history logits screening. Its teacher-forced generated tokens are equal by construction and are NOT independent token-parity evidence. No native-sparse serving speedup is claimed.

## Fresh serving comparison

H200 GPUs 4–7, TP4/EP1/DP1, BS1, both MTP5, BF16 KV, FP32 recurrent state, max batch16, prefill chunk2048, session270336, output5, greedy/seed42, prefix off. Sequential fresh services, 3 warmups +30 measured requests per length/engine, profiler inactive. Independent medians; TPOT=(last choices response−first choices response)/4 per request.

| Input | Metric | LMDeploy | vLLM |
|---:|---|---:|---:|
| 2,048 | TTFT | 234.60 ms | 138.45 ms |
| 2,048 | TPOT | 7.28 ms | 7.24 ms |
| 2,048 | Total request | 264.15 ms | 167.49 ms |
| 8,192 | TTFT | 892.82 ms | 557.40 ms |
| 8,192 | TPOT | 7.08 ms | 6.68 ms |
| 8,192 | Total request | 921.83 ms | 582.79 ms |

## Separate CPU+CUDA diagnostics

These profiled requests include collection overhead and do not replace the serving table. Rank0/trial0 is used consistently. Client clock samples are overlaid after capture; CPU forward scopes, GPU verification spans and client TPOT are different metrics.

| Engine | Input | TTFT | Client post-first-response window | TPOT (window / 4) | Second target verification GPU span |
|---|---:|---:|---:|---:|---:|
| LMDeploy | 2,048 | 320.019 ms | 39.725 ms | 9.931 ms | 13.345 ms |
| LMDeploy | 8,192 | 1224.736 ms | 40.801 ms | 10.200 ms | 13.595 ms |
| vLLM | 2,048 | 186.481 ms | 38.247 ms | 9.562 ms | 8.438 ms |
| vLLM | 8,192 | 727.148 ms | 36.988 ms | 9.247 ms | 8.448 ms |

Raw four-rank traces: final_compare/torch_lmdeploy and final_compare/torch_vllm. Timestamp-overlay copies: final_compare/torch_lmdeploy_with_client and final_compare/torch_vllm_with_client. Open those copies in https://ui.perfetto.dev/ and select `decode_window / 4 = TPOT`; its Duration/4 equals the same request's TPOT. Work submitted after the last client response is excluded from that client window.
