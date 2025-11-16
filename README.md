# Radix Attention Homework

This repository contains my RadixAttention modifications to `llama.cpp` plus the automation I used to benchmark Gemma 3 4B on **llava-bench-in-the-wild**. The notes below document exactly how to fetch the model weights, build the server, and run the benchmark scripts.

> **Prereqs**
> - macOS with Xcode CLT (or another toolchain supported by `llama.cpp`)
> - `cmake`, `ninja` (or make), and `huggingface_hub` (`pip install huggingface-hub`)
> - A Hugging Face token with access to `google/gemma-3-4b-it`

---

## 1. Download the models

```bash
cd /Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4

# Optional cache directory keeps repeated downloads fast
export HF_HOME=hf-cache

# Quantized text weights (q4_K_M) and Gemma 3 multimodal projector
huggingface-cli download google/gemma-3-4b-it \
  --local-dir models/gemma3-4b \
  --include "gemma-3-4b-it-text-q4_K_M.gguf" \
  --cache-dir "${HF_HOME}"

huggingface-cli download google/gemma-3-4b-it \
  --local-dir models/gemma3-4b \
  --include "mmproj-model-f16-4B.gguf" \
  --cache-dir "${HF_HOME}"
```

The `.gitignore` already excludes `models/` and `hf-cache/`, so these artifacts stay local-only.

---

## 2. Build `llama-server`

```bash
cd /Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4/llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --target llama-server
```

On Apple Silicon this enables the Metal backend automatically (`-DGGML_METAL=ON`). Adjust the flags to match your GPU/CPU environment if you build elsewhere.

---

## 3. Run the RadixAttention benchmark

The helper script does four things: kills any stray `llama-server`, launches a fresh server with the RadixAttention flags, blocks until `/health` is ready, and finally runs `scripts/run_llava_bench.py`.

```bash
cd /Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4
./run_radix_benchmark.sh
```

Artifacts land under `artifacts/radix/`:

- `server.log` – full server trace for the run
- `gemma3-4b-radix.json` – llava-bench metrics (latency, similarity, etc.)

If you want a baseline (no RadixAttention), run `scripts/run_llava_bench.py` manually against a clean `llama-server` invocation without `--cache-reuse` and compare against `artifacts/baseline/gemma3-4b-baseline.json`.

---

## 4. Manual server launch (optional)

Use this when you want to poke the HTTP API yourself:

```bash
cd /Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4
./llama.cpp/build/bin/llama-server \
  -m models/gemma3-4b/gemma-3-4b-it-text-q4_K_M.gguf \
  --mmproj models/gemma3-4b/mmproj-model-f16-4B.gguf \
  --alias gemma-3-4b-it-q4_K_M \
  --port 8080 \
  --cache-reuse 256 \
  --ctx-size 4096 \
  -ngl 99 \
  --ubatch-size 512 \
  --batch-size 1024 \
  --log-verbosity 0
```

Once you see `server is listening on http://127.0.0.1:8080`, send OpenAI-compatible requests to `localhost:8080` (e.g., `curl http://127.0.0.1:8080/v1/chat/completions ...`) or rerun the benchmark script.

---

## 5. Results

- Baseline metrics live in `artifacts/baseline/gemma3-4b-baseline.json`
- RadixAttention metrics live in `artifacts/radix/gemma3-4b-radix.json`
- `design.md` summarizes the architecture changes and compares those two JSON files.

Use these as the canonical references when writing reports or reproducing the experiments. If you regenerate numbers, remember to copy the refreshed JSON into `artifacts/radix/` (or `baseline/`) before you recompute deltas.

