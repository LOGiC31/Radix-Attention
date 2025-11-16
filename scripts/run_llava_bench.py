#!/usr/bin/env python3
import argparse
import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
from datasets import load_dataset
from difflib import SequenceMatcher
from tqdm import tqdm


@dataclass
class SampleResult:
    question_id: int
    image_id: str
    category: str
    question: str
    reference: str
    response: str
    latency_ms: float
    similarity: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "image_id": self.image_id,
            "category": self.category,
            "question": self.question,
            "reference": self.reference,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "similarity": self.similarity,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def pil_image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def run_benchmark(
    server_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    limit: Optional[int],
    output_path: Path,
) -> None:
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")
    records: List[SampleResult] = []

    for row in tqdm(dataset, desc="Evaluating", total=(limit or len(dataset))):
        if limit is not None and len(records) >= limit:
            break

        data_url = pil_image_to_data_url(row["image"])
        payload = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": row["question"]},
                    ],
                }
            ],
        }

        start = time.perf_counter()
        response = requests.post(f"{server_url}/v1/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        payload_out = response.json()

        choice = payload_out["choices"][0]
        answer = choice["message"]["content"].strip()

        usage = payload_out.get("usage") or {}
        similarity = SequenceMatcher(
            None,
            answer.lower(),
            (row["gpt_answer"] or "").lower(),
        ).ratio()

        records.append(
            SampleResult(
                question_id=int(row["question_id"]),
                image_id=row["image_id"],
                category=row["category"],
                question=row["question"],
                reference=row["gpt_answer"],
                response=answer,
                latency_ms=latency_ms,
                similarity=similarity,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        )

    output = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "server_url": server_url,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_samples": len(records),
        "avg_latency_ms": sum(r.latency_ms for r in records) / max(len(records), 1),
        "avg_similarity": sum(r.similarity for r in records) / max(len(records), 1),
        "results": [r.to_dict() for r in records],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {output_path}")
    print(f"Average latency: {output['avg_latency_ms']:.1f} ms over {len(records)} samples")
    print(f"Average similarity: {output['avg_similarity']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaVA-bench with llama.cpp server")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Base URL for llama-server")
    parser.add_argument(
        "--model",
        default="gemma-3-4b-it-q4_K_M",
        help="Model name/alias exposed by llama-server",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baseline/gemma3-4b-baseline.json"),
        help="Path to write JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        server_url=args.server,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
