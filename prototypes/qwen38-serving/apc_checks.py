"""Bounded align-APC hit and shared-prefix witnesses, outside service timing.

Use through elastic_probe with ELASTIC_APC=1 and ELASTIC_SHADOW=1.
The pinned Qwen27/Ascend layout has 1536-token cache blocks, so a 513-token
prompt cannot qualify reuse. Cached-token receipts are mandatory witnesses.
"""

import concurrent.futures
import json
import urllib.request
from service_probe import request


def run(url, prompt, prompts, arm, check_shadow, receipt, path):
    receipt["apc_checks"] = []

    def cached(row):
        return row["usage"]["prompt_tokens_details"]["cached_tokens"]

    def reset():
        req = urllib.request.Request(url + "/reset_prefix_cache", method="POST")
        with urllib.request.urlopen(req, timeout=60) as response:
            assert response.status == 200

    for n in (1537, 2051, 3073):
        tokens = (prompt + prompt)[:n]
        assert len(tokens) == n
        reset()
        cold = request(url, tokens, 8)
        arm(2)
        warm = request(url, tokens, 8)
        check_shadow()
        # Serving comparison intentionally does not demand bitwise native output;
        # this small controlled greedy witness should still reproduce the text.
        assert cached(cold) == 0, cold["usage"]
        assert cached(warm) > 0, warm["usage"]
        assert cold["text"] == warm["text"], (n, cold["text"], warm["text"])
        receipt["apc_checks"].append(dict(length=n, cold=cold, warm=warm))
        path.write_text(json.dumps(receipt, indent=2))
    # Multiple users share a cached prefix but update distinct running states.
    shared = prompts[2051]
    branches = [shared + [33 + i] * (17 + i) for i in range(8)]
    cold_branches = []
    for tokens in branches:
        reset()
        cold_branches.append(request(url, tokens, 8))
    reset()
    request(url, shared, 8)
    arm(6)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        warm_branches = list(pool.map(lambda tokens: request(url, tokens, 8), branches))
    check_shadow()
    for cold, warm in zip(cold_branches, warm_branches):
        assert cached(warm) > 0, warm["usage"]
        assert cold["text"] == warm["text"], (cold["text"], warm["text"])
    receipt["apc_branches"] = dict(cold=cold_branches, warm=warm_branches)
