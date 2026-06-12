"""Exp-6b Phase 0 smoke test: assert LM Studio returns logprobs."""
import requests

r = requests.post("http://127.0.0.1:1234/v1/chat/completions", json={
    "model": "local-model", "messages": [{"role": "user", "content": "Say A or B."}],
    "max_tokens": 3, "logprobs": True, "top_logprobs": 5}, timeout=60)
lp = r.json()["choices"][0]["logprobs"]
assert lp and lp["content"] and len(lp["content"][0]["top_logprobs"]) >= 2, lp
print("OK: logprobs present.", [t["token"] for t in lp["content"][0]["top_logprobs"]])
