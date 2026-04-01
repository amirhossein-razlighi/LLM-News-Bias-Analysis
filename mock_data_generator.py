import requests
import random
import time
from datetime import datetime

URL = "http://127.0.0.1:8000/ingest"


def generate_and_send(num_incidents=15):
    payload = []
    models = ["openai/gpt-4o-mini", "ollama/llama3"]
    conditions = ["headlines_only", "headlines_sources", "sources_only", "swapped_sources"]
    buckets = ["left", "center", "right"]

    for i in range(num_incidents):
        inc_id = f"INC_{100 + i}"
        model = random.choice(models)

        for cond in conditions:
            # Random logic: make llama3 slightly more 'left' biased for testing
            if model == "ollama/llama3":
                pick = random.choices(buckets, weights=[50, 30, 20])[0]
            else:
                pick = random.choice(buckets)

            result = {
                "request_id": f"REQ_{inc_id}_{cond}_{model.replace('/', '_')}",
                "incident_id": inc_id,
                "condition": cond,
                "model_name": model,
                "selected_article_id": f"ART_{pick}_{i}",
                "selected_outlet": f"{pick.capitalize()} News Network",
                "selected_bucket": pick,
                "justification": "Automated mock justification.",
                "raw_response": "Full raw text here...",
                "parsed_successfully": True,
                "latency_ms": random.randint(400, 1500),
                "timestamp_utc": {datetime.utcnow().timestamp()}
            }
            payload.append(result)

    try:
        res = requests.post(URL, json=payload)
        print(f"Status: {res.status_code} | Data Sent: {len(payload)} records.")
    except Exception as e:
        print(f"Error: Could not connect to the server. Is member2_analytics_engine.py running? \n{e}")


if __name__ == "__main__":
    generate_and_send()