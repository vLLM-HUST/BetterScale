"""Admitted public CLI/HTTP vertical; owns and reclaims only its service child."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def main():
    run = Path(os.environ["CAPSULE"])
    import importlib.util

    origin = Path(importlib.util.find_spec("betterscale").origin).resolve()
    assert origin.is_relative_to(Path(os.environ["INSTALLED_PACKAGE"]).resolve()), (
        origin
    )
    port = int(os.environ.get("PROBE_PORT", "18935"))
    model = Path("/workspace/models/Qwen3.5-35B-A3B")
    reference = json.loads(Path(os.environ["NATIVE_REFERENCE"]).read_text())
    command = [
        sys.executable,
        "-m",
        "betterscale",
        "serve-qwen",
        str(model),
        "--runtime",
        "live",
        "--devices",
        "0,1",
        "--port",
        str(port),
        "--cache-dir",
        str(run / "service-cache"),
        "--live-context-tokens",
        "512",
        "--live-resident-seats",
        "20",
        "--live-token-pages",
        "64",
        "--live-distributed-port",
        "29637",
    ]
    log_path = run / "service.log"
    receipts = {}

    def request(path, payload=None):
        req = Request(
            f"http://127.0.0.1:{port}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=180) as response:
            return json.load(response)

    def retain(key, value):
        receipts[key] = value
        (run / "http-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps({"case": key, "result": value}), flush=True)
        return value

    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"live service exited {process.returncode}")
                try:
                    health = request("/health")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("live service readiness deadline")
                    time.sleep(1)
            retain("health", health)
            assert health["runtime"] == "live" and health["root"] == "QwenLiveLLMRoot"
            assert set(health["graphs"]) == {"target1", "target3", "draft1", "draft2"}
            assert health["resident_seats"] == 20
            retain("models", request("/v1/models"))
            for i, item in enumerate(reference["chats"]):
                result = retain(
                    f"chat{i}",
                    request(
                        "/v1/chat/completions",
                        {
                            "messages": [{"role": "user", "content": item["question"]}],
                            "max_tokens": 16,
                            "temperature": 0,
                        },
                    ),
                )
                assert result["betterscale"]["token_ids"] == item["token_ids"], (
                    "live/native chat differs",
                    i,
                    result["betterscale"]["token_ids"],
                    item["token_ids"],
                )
            base = reference["base"]
            a = retain(
                "raw",
                request(
                    "/v1/completions",
                    {"prompt": base["prompt"], "max_tokens": 12, "ignore_eos": True},
                ),
            )
            assert a["betterscale"]["token_ids"] == base["token_ids"], (
                "raw live/native mismatch"
            )
            b = retain(
                "unrelated",
                request(
                    "/v1/completions",
                    {"prompt": [16, 10, 17, 28], "max_tokens": 4, "ignore_eos": True},
                ),
            )
            assert b["betterscale"]["seat"] != a["betterscale"]["seat"]
            prompt = base["prompt"] + a["betterscale"]["token_ids"] + [271]
            c = retain(
                "warm",
                request(
                    "/v1/completions",
                    {"prompt": prompt, "max_tokens": 8, "ignore_eos": True},
                ),
            )
            assert c["betterscale"]["seat"] == a["betterscale"]["seat"]
            assert c["betterscale"]["cached_tokens"] == len(prompt) - 1
            long_message = (
                "Context: The meeting is on Monday. " * 20
                + "\nRepeat exactly this code and nothing else: ORCHID-7319"
            )
            long_payload = {
                "messages": [{"role": "user", "content": long_message}],
                "max_tokens": 16,
            }
            first_long = retain(
                "long_chat", request("/v1/chat/completions", long_payload)
            )
            assert first_long["usage"]["prompt_tokens"] > 128
            assert first_long["choices"][0]["message"]["content"] == "ORCHID-7319"
            repeat_long = retain(
                "long_chat_repeat", request("/v1/chat/completions", long_payload)
            )
            assert (
                repeat_long["betterscale"]["token_ids"]
                == first_long["betterscale"]["token_ids"]
            )
            assert repeat_long["betterscale"]["cached_tokens"] == 0, (
                "later State used as earlier prefix"
            )
            try:
                request("/v1/completions", {"prompt": [1, 2], "temperature": 1})
            except HTTPError as error:
                assert error.code == 400
                retain("unsupported", json.loads(error.read()))
            else:
                raise AssertionError("non-greedy request accepted")
            retain("health_after_rejection", request("/health"))
        finally:
            # Uvicorn's rank0 PID is logged by the service. Signal only that
            # descendant so its final control message can release the other rank.
            import psutil

            descendants = (
                {p.pid for p in psutil.Process(process.pid).children(recursive=True)}
                if process.poll() is None
                else set()
            )
            matches = re.findall(
                r"Started server process \[(\d+)\]", log_path.read_text()
            )
            if matches and int(matches[-1]) in descendants:
                os.kill(int(matches[-1]), signal.SIGINT)
            elif process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            retain("service_exit", process.returncode)
    assert process.returncode == 0


if __name__ == "__main__":
    main()
