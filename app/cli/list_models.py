from __future__ import annotations

import argparse
import json

from app.models.ollama_client import OllamaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List locally available Ollama models")
    parser.add_argument("--base-url", default="http://localhost:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OllamaClient(base_url=args.base_url)
    models = client.list_models()
    print(json.dumps({"models": models}, indent=2))


if __name__ == "__main__":
    main()
