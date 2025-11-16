#!/bin/bash

# Script to run the benchmark with radix attention enabled

MODEL_PATH="/Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4/models/gemma3-4b/gemma-3-4b-it-text-q4_K_M.gguf"
MMPROJ_PATH="/Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4/models/gemma3-4b/mmproj-model-f16-4B.gguf"
SERVER_BIN="/Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4/llama.cpp/build/bin/llama-server"
SERVER_HOST="http://127.0.0.1:8080"
OUTPUT_DIR="/Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4/artifacts/radix"
OUTPUT_FILE="${OUTPUT_DIR}/gemma3-4b-radix.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Ensure no previous llama-server instance is running (prevents port conflicts / log mixups)
EXISTING_PIDS=$(pgrep -f "${SERVER_BIN}")
if [ -n "${EXISTING_PIDS}" ]; then
    echo "Found existing llama-server processes: ${EXISTING_PIDS}"
    echo "Stopping previous llama-server instances..."
    # Use xargs to handle multiple PIDs in a single call
    echo "${EXISTING_PIDS}" | xargs kill 2>/dev/null
    # Give the OS a moment to reap old processes
    sleep 2
fi

# Start the server in the background with cache-reuse enabled (radix attention)
# NOTE: use low log verbosity so we don't spam server.log in case of a stuck loop
echo "Starting llama-server with radix attention (cache-reuse=256)..."
"${SERVER_BIN}" \
    -m "${MODEL_PATH}" \
    --mmproj "${MMPROJ_PATH}" \
    --alias "gemma-3-4b-it-q4_K_M" \
    --port 8080 \
    --cache-reuse 256 \
    --ctx-size 4096 \
    -ngl 99 \
    --ubatch-size 512 \
    --batch-size 1024 \
    --log-verbosity -1 \
    > "${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "Server started with PID: ${SERVER_PID}"
echo "Waiting for server to be ready..."

WAIT_TIMEOUT=1
WAIT_INTERVAL=2
WAIT_ELAPSED=0
HEALTH_ENDPOINT="${SERVER_HOST}/health"

until curl -sSf "${HEALTH_ENDPOINT}" >/dev/null 2>&1; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "Error: Server exited while starting up. Check ${OUTPUT_DIR}/server.log"
        exit 1
    fi
    if [ ${WAIT_ELAPSED} -ge ${WAIT_TIMEOUT} ]; then
        echo "Error: Server health check timed out after ${WAIT_TIMEOUT}s. See ${OUTPUT_DIR}/server.log"
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
        exit 1
    fi
    sleep ${WAIT_INTERVAL}
    WAIT_ELAPSED=$((WAIT_ELAPSED + WAIT_INTERVAL))
    echo "Trying to connect to server... ${WAIT_ELAPSED}s"
done

echo "Server is healthy after ${WAIT_ELAPSED}s"

# Run the benchmark
echo "Running benchmark..."
cd /Users/vinaysingh/Desktop/deskkk/csce689-llm/hw4

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

python3 scripts/run_llava_bench.py \
    --server "${SERVER_HOST}" \
    --model "gemma-3-4b-it-q4_K_M" \
    --temperature 0.2 \
    --max-tokens 256 \
    --output "${OUTPUT_FILE}"

BENCHMARK_EXIT_CODE=$?

# Stop the server
echo "Stopping server..."
kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null

if [ ${BENCHMARK_EXIT_CODE} -eq 0 ]; then
    echo "Benchmark completed successfully!"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
    echo "Comparing with baseline..."
    if [ -f "artifacts/baseline/gemma3-4b-baseline.json" ]; then
        python3 -c "
import json
with open('artifacts/baseline/gemma3-4b-baseline.json') as f:
    baseline = json.load(f)
with open('${OUTPUT_FILE}') as f:
    radix = json.load(f)
print(f'Baseline - Avg Latency: {baseline[\"avg_latency_ms\"]:.2f} ms, Avg Similarity: {baseline[\"avg_similarity\"]:.4f}')
print(f'Radix    - Avg Latency: {radix[\"avg_latency_ms\"]:.2f} ms, Avg Similarity: {radix[\"avg_similarity\"]:.4f}')
print(f'Latency Improvement: {((baseline[\"avg_latency_ms\"] - radix[\"avg_latency_ms\"]) / baseline[\"avg_latency_ms\"] * 100):.2f}%')
print(f'Similarity Change: {((radix[\"avg_similarity\"] - baseline[\"avg_similarity\"]) / baseline[\"avg_similarity\"] * 100):.2f}%')
"
    fi
else
    echo "Benchmark failed with exit code: ${BENCHMARK_EXIT_CODE}"
    exit ${BENCHMARK_EXIT_CODE}
fi

