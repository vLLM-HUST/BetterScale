"""Narrow loopback HTTP ingress; one completed-wave execution owner.

Only greedy text completion/chat is admitted. Unsupported serving features are
rejected, not silently approximated or delegated to a native engine.
"""

import asyncio
import inspect
import json
import time
import uuid
from pathlib import Path


def request_tokens(payload, tokenizer, *, chat, context_tokens, vocab_size):
    allowed = {
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "ignore_eos",
        "return_token_ids",
        "stream_options",
        "cache_salt",
        "seed",
    }
    allowed.add("messages" if chat else "prompt")
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("unsupported live parameters: " + ", ".join(sorted(unknown)))
    if (
        payload.get("temperature", 0) != 0
        or payload.get("top_p", 1) != 1
        or payload.get("n", 1) != 1
    ):
        raise ValueError("live entry supports greedy, n=1 only")
    for name in ("stream", "return_token_ids"):
        if type(payload.get(name, False)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if chat and payload.get("stream", False):
        raise ValueError("streaming currently requires the completions endpoint")
    options = payload.get("stream_options", {})
    if (
        not isinstance(options, dict)
        or set(options) - {"include_usage"}
        or type(options.get("include_usage", False)) is not bool
    ):
        raise ValueError("unsupported stream_options")
    if "seed" in payload and type(payload["seed"]) is not int:
        raise ValueError(
            "seed must be an integer; greedy execution does not use randomness"
        )
    if "cache_salt" in payload and (
        not isinstance(payload["cache_salt"], str)
        or not 1 <= len(payload["cache_salt"]) <= 256
    ):
        raise ValueError(
            "cache_salt must be a nonempty string of at most 256 characters"
        )
    count = payload.get("max_tokens", 16)
    if type(count) is not int or count <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if chat:
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or m["role"] not in ("user", "assistant", "system")
                or not isinstance(m["content"], str)
                for m in messages
            )
        ):
            raise ValueError("messages must contain plain role/content text pairs")
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt = payload.get("prompt")
        tokens = (
            tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str)
            else prompt
        )
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)
    ):
        raise ValueError("prompt must be nonempty text or valid token IDs")
    if len(tokens) + count > context_tokens:
        raise ValueError("prompt and output exceed the live context envelope")
    if type(payload.get("ignore_eos", False)) is not bool:
        raise ValueError("ignore_eos must be boolean")
    return tokens, count


def create_app(root, tokenizer, model_name, execute, *, eos_token_ids=None):
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(title="BetterScale experimental live runtime")
    eos = (
        root.target_model.model.config.eos_token_id
        if eos_token_ids is None
        else eos_token_ids
    )
    eos = (eos,) if isinstance(eos, int) else tuple(eos or ())

    @app.get("/health")
    async def health():
        if getattr(root, "serving_error", None) is not None:
            raise HTTPException(503, "live execution owner failed")
        return {
            "status": "ok",
            "runtime": "live",
            "root": type(root).__name__,
            "graphs": [name for name, _ in root.named_graphs()],
            "context_tokens": root.context_tokens,
            "state_capacity": getattr(
                getattr(getattr(root, "_live_runtime", None), "_state_backend", None),
                "capacity_report", None,
            ),
            "resident_seats": root.capacity.resident_seats,
            "execution_seats": getattr(root.capacity, "execution_seats", 1),
            "scheduler": root.scheduler.snapshot()
            if hasattr(root, "scheduler")
            else None,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "betterscale"}],
        }

    async def complete(payload, chat, request):
        if payload.get("model", model_name) != model_name:
            raise HTTPException(400, "model does not match this live deployment")
        try:
            tokens, count = request_tokens(
                payload,
                tokenizer,
                chat=chat,
                context_tokens=root.context_tokens,
                vocab_size=root.target_model.model.vocab_size,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        stops = () if payload.get("ignore_eos", False) else eos
        execution_options = (
            {"cache_salt": payload["cache_salt"]} if "cache_salt" in payload else {}
        )
        if payload.get("stream", False):
            from starlette.responses import StreamingResponse

            async def events():
                pending, output = [], []
                changed = asyncio.Event()
                identifier = "cmpl-" + uuid.uuid4().hex
                created = int(time.time())
                text_sent = ""

                def receive(ids):
                    pending.extend(ids)
                    changed.set()

                def event(choices, **extra):
                    body = dict(
                        id=identifier,
                        created=created,
                        model=model_name,
                        object="text_completion",
                        choices=choices,
                        **extra,
                    )
                    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"

                task = asyncio.create_task(
                    execute(
                        tokens, count, stops, on_tokens=receive, **execution_options
                    )
                )
                task.add_done_callback(lambda _: changed.set())
                try:
                    first = True
                    while True:
                        await changed.wait()
                        changed.clear()
                        if pending:
                            delta = list(pending)
                            pending.clear()
                            output.extend(delta)
                            decoded = tokenizer.decode(
                                output,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                            text = ""
                            if not decoded.endswith("\ufffd"):
                                if not decoded.startswith(text_sent):
                                    raise RuntimeError(
                                        "tokenizer rewrote already streamed text"
                                    )
                                text, text_sent = decoded[len(text_sent) :], decoded
                            choice = dict(index=0, text=text, finish_reason=None)
                            if payload.get("return_token_ids", False):
                                choice["token_ids"] = delta
                                if first:
                                    choice["prompt_token_ids"] = tokens
                            first = False
                            yield event([choice])
                        if task.done():
                            break
                    result = await task
                    if output != result["token_ids"]:
                        raise RuntimeError("stream differs from committed output")
                    decoded = tokenizer.decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if not decoded.startswith(text_sent):
                        raise RuntimeError("tokenizer rewrote final streamed text")
                    yield event(
                        [
                            dict(
                                index=0,
                                text=decoded[len(text_sent) :],
                                token_ids=[],
                                finish_reason="stop"
                                if output[-1] in stops
                                else "length",
                            )
                        ]
                    )
                    if payload.get("stream_options", {}).get("include_usage", False):
                        yield event(
                            [],
                            usage=dict(
                                prompt_tokens=len(tokens),
                                completion_tokens=len(output),
                                total_tokens=len(tokens) + len(output),
                                prompt_tokens_details={
                                    "cached_tokens": result["cached_tokens"]
                                },
                            ),
                        )
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    yield (
                        "data: "
                        + json.dumps(
                            {"error": {"message": str(exc), "type": type(exc).__name__}}
                        )
                        + "\n\n"
                    )
                finally:
                    if not task.done():
                        task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

            return StreamingResponse(events(), media_type="text/event-stream")
        try:
            result = execute(tokens, count, stops, **execution_options)
            if inspect.isawaitable(result):
                task = asyncio.create_task(result)
                try:
                    while not task.done():
                        await asyncio.wait({task}, timeout=0.1)
                        if not task.done() and await request.is_disconnected():
                            raise HTTPException(499, "client disconnected")
                    result = await task
                finally:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        output = result["token_ids"]
        text = tokenizer.decode(output, skip_special_tokens=True)
        choice = {
            "index": 0,
            "finish_reason": "stop" if output[-1] in stops else "length",
        }
        choice.update(
            {"message": {"role": "assistant", "content": text}}
            if chat
            else {"text": text}
        )
        if payload.get("return_token_ids", False):
            choice.update(token_ids=output, prompt_token_ids=tokens)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(output),
                "total_tokens": len(tokens) + len(output),
                "prompt_tokens_details": {"cached_tokens": result["cached_tokens"]},
            },
            "betterscale": {**result, "runtime": "live"},
        }

    @app.post("/v1/completions")
    async def completions(payload: dict, request: Request):
        return await complete(payload, False, request)

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict, request: Request):
        return await complete(payload, True, request)

    return app


def run_owned_http(app, *, port):
    """Consume shutdown signals here; bootstrap must retire the model afterward.

    Native event loops may return None for a C-installed handler. Python cannot
    restore that value. Restore the process default in that case, and do not
    re-raise the graceful shutdown signal before distributed State retirement.
    """
    import signal
    import threading
    from contextlib import contextmanager

    import uvicorn

    class LiveServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("live HTTP owns process signals on the main thread")
            previous = {
                sig: signal.signal(sig, self.handle_exit)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                yield
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, signal.SIG_DFL if handler is None else handler)

    LiveServer(uvicorn.Config(app, host="127.0.0.1", port=port)).run()


def serve(root, model_path, *, port):
    import torch
    from transformers import AutoTokenizer
    from vllm.distributed import get_world_group

    group = get_world_group().cpu_group
    rank = torch.distributed.get_rank()
    from .ingress import Ingress
    from .scheduler import Scheduler

    if rank:
        scheduler = Scheduler(root)
        try:
            while True:
                message = [None]
                torch.distributed.broadcast_object_list(message, src=0, group=group)
                if message[0] is None:
                    return
                for command in message[0]:
                    if command[0] == "submit":
                        scheduler.submit(*command[1:])
                    else:
                        scheduler.cancel(command[1])
                scheduler.tick()
        finally:
            scheduler.close()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        def broadcast(commands):
            torch.distributed.broadcast_object_list([commands], src=0, group=group)

        ingress = Ingress(root, broadcast)
        generation_path = Path(model_path) / "generation_config.json"
        eos = (
            json.loads(generation_path.read_text()).get("eos_token_id")
            if generation_path.exists()
            else None
        )
        app = create_app(
            root, tokenizer, Path(model_path).name, ingress.execute, eos_token_ids=eos
        )
        app.router.lifespan_context = ingress.lifespan
        try:
            run_owned_http(app, port=port)
        finally:
            torch.distributed.broadcast_object_list([None], src=0, group=group)
