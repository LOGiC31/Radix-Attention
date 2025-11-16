# RadixAttention Design & Benchmark Report
#### Vinay Singh - 335008079
## 1. Deployment & Benchmark Setup

- **Model**: `gemma-3-4b-it-q4_K_M.gguf` with `mmproj-model-f16-4B.gguf`
- **Serving stack**: `llama.cpp` `llama-server` on Apple M2 (Metal backend), `--cache-reuse 256`, `--ctx-size 4096`, `-ngl 99`, `--ubatch-size 512`, `--batch-size 1024`
- **Client driver**: `scripts/run_llava_bench.py` (LLaVA Bench in the Wild, 60-image eval, temperature 0.2, max 256 tokens)
- **Artifacts compared**:
  - Baseline prompt-cache only: `artifacts/baseline/gemma3-4b-baseline.json`
  - RadixAttention run: `artifacts/radix/gemma3-4b-radix.json`

## 2. RadixAttention Implementation Summary

Key changes live in `llama.cpp/tools/server/server.cpp` with supporting knobs in `common/common.h` and CLI plumbing in `common/arg.cpp`.

### 2.1 Slot-local Radix Cache

- Each `server_slot` now owns a `radix_tree`, a filtered `radix_text_tokens` buffer, `radix_text_to_full` map, and TTL bookkeeping (`radix_last_rebuild_us`).
- `radix_rebuild_from_prompt()` compacts the prompt to text-only tokens (skip LLAMA_TOKEN_NULL multimodal sentinels), trims to a configurable tail (`cache_reuse_max_tokens`), and rebuilds the trie when the slot is released or after cache mutation.

### 2.2 Matching Policy & Safety

- `radix_attention_config` captures tunables: minimum chunk size, max matches per request, query window, coverage threshold, TTL, and max cached tokens.
- Before reuse:
  - Convert the new request’s suffix into text-only form and align indices back to the full multimodal prompt.
  - Drop the trie when it expires (TTL) or is empty.
  - Collect non-overlapping candidates via `radix_tree::find_all_matches`, then verify token equality before shifting KV cache blocks (`llama_memory_seq_rm/add`).
  - Guard `server_tokens::set_token` calls so we never overwrite LLAMA_TOKEN_NULL slots (prevents the MTMD assertion seen earlier).
- If no valid radix reuse is found, fall back to the legacy linear matching loop.

### 2.3 CLI Controls

New server flags (defaults in `common/common.h`):

- `--cache-reuse-ttl SECONDS`
- `--cache-reuse-min-overlap RATIO`
- `--cache-reuse-max-matches N`
- `--cache-reuse-max-tokens N`

### 2.4 Operational Guardrails

- `run_radix_benchmark.sh` now:
  - Kills stray `llama-server` PIDs before launching a fresh one.
  - Waits for the `/health` endpoint (up to 180 s) instead of sleeping blindly.

### 2.5 Source-level Diff Highlights

- `tools/server/server.cpp`
  - Added the `radix_attention_config` struct plus `server_context::radix_cfg`, `configure_radix_attention()`, and helper logic to propagate CLI knobs into each slot.
  - Extended `server_slot` with `radix_tree radix_cache`, token remap vectors, TTL bookkeeping, and lifecycle helpers (`radix_clear`, `radix_rebuild_from_prompt`).
  - Reworked the prompt-reuse branch inside `update_slots()` so we:
    - Extract text-only subsequences from multimodal prompts.
    - Guard token replacement when `has_mtmd` is true (fixes the `tokens[pos] != LLAMA_TOKEN_NULL` assertion).
    - Score radix matches, enforce coverage thresholds, and shift KV cache ranges through `llama_memory_seq_rm/add`.
    - Fall back to the legacy linear matcher when the trie is empty or disabled.
  - Tightened slot-reset paths (`server_slot::reset`, `server_slot::release`) to rebuild or drop radix data at deterministic times.
- `common/common.h` / `common/arg.cpp`
  - Introduced the persistent parameters backing the new `--cache-reuse-*` flags and plumbed them through `common_arg` definitions.
  - Defaulted the new knobs (TTL, max matches, min overlap, max query tokens) so existing users get conservative behavior without extra flags.
- `run_radix_benchmark.sh`
  - Added a `pgrep`/`kill` preamble to avoid stacking multiple `llama-server` instances.
  - Replaced the fixed `sleep` with a curl-based `/health` poll that times out cleanly and surfaces startup failures early.

## 3. Benchmark Results

| Variant        | Samples | Avg Latency (ms) | Avg Similarity |
| -------------- | ------- | ---------------- | -------------- |
| Baseline       | 60      | 15,314.02        | 0.1219         |
| RadixAttention | 60      | 15,962.49        | 0.0952         |

_(Source: `artifacts/baseline/gemma3-4b-baseline.json` & `artifacts/radix/gemma3-4b-radix.json`)_

## 4. Metric Analysis

### 4.1 Latency Regression (~+4.2%)

1. **Trie build overhead per slot**: We rebuild the radix tree whenever a slot releases or its prompt is mutated. For multimodal prompts (all 60 are image+text), each prompt includes long LLAMA_TOKEN_NULL stretches; filtering and remapping adds CPU time while yielding few reusable spans.
2. **Low reuse yield**: The evaluation set rarely repeats the same textual tail, so the radix matcher often aborts after scanning 512 tokens without shifting any KV blocks—cost with no payoff.
3. **TTL-induced rebuilds**: Default TTL (900 s) is generous, but the benchmark spans ~25 minutes. Slots recycled late in the run frequently exceeded the TTL and rebuilt from scratch, eliminating any accumulated benefit.
4. **Extra logging**: Verbose `SLT_INF` traces for debugging (match previews, coverage stats) still run even with `--log-verbosity -1`, marginally increasing host overhead.

Result: the added preprocessing + matching passes more than offset the modest KV savings under this workload, producing slower end-to-end latencies.

### 4.2 Similarity Drop (~22% relative)

1. **Aggressive chunk shifting with mixed modalities**: Even with the LLAMA_TOKEN_NULL guard, aligned spans can splice text around image tokens differently than the original prompt tokens, causing slightly altered context windows and downstream answers.
2. **Coverage threshold default (0.25)**: Small matches that pass the ratio test may still replace critical question text with cached tokens captured from unrelated conversations, injecting stale context.
3. **No domain-specific warm cache**: Because each benchmark request is independent, reusing prior prompts can actually hurt if the reused text relates to a different image. Similarity falls when hallucinated carry-over facts appear in answers.
