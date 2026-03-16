from __future__ import annotations

import argparse
import json
import statistics

from app.models.ollama_client import OllamaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare latency across Ollama models")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--base-url", default="http://localhost:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OllamaClient(base_url=args.base_url)

    report = []
    for model in args.models:
        latencies = []
        for _ in range(args.rounds):
            generation = client.generate(model=model, prompt=args.prompt)
            latencies.append(generation.latency_ms)
        report.append(
            {
                "model": model,
                "rounds": args.rounds,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "mean_latency_ms": int(statistics.mean(latencies)),
            }
        )

    print(json.dumps({"results": report}, indent=2))


if __name__ == "__main__":
    main()
