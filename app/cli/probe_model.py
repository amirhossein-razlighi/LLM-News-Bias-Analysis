from __future__ import annotations

import argparse
import json

from app.experiment.prompt_builder import selection_response_json_schema
from app.models.ollama_client import OllamaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single prompt against one Ollama model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--think", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OllamaClient(base_url=args.base_url)
    generation = client.generate(
        model=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        think=args.think,
        response_schema=selection_response_json_schema(),
    )
    print(
        json.dumps(
            {
                "latency_ms": generation.latency_ms,
                "response": generation.text,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
