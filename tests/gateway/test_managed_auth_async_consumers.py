"""Blocking managed auth must not stall the gateway loop at async consumers."""
import asyncio
import ast
import inspect
import threading
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["context", "image", "background", "hygiene_metadata", "hygiene_compression"])
async def test_async_consumers_keep_loop_responsive_during_managed_auth(monkeypatch, caller):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig()
    key = runner._session_key_for_source(source)
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_model_overrides = {key: {
        "model": "gpt-5.4", "provider": "openai-codex", "api_key": "stale-test-token",
        "_managed_credentials": True,
    }}
    entered, release = threading.Event(), threading.Event()
    timed_out = []
    threads = []

    def blocking_resolution(provider):
        threads.append(threading.get_ident())
        entered.set()
        if not release.wait(0.5):
            timed_out.append(True)
        return {"provider": provider, "api_key": "" if caller == "background" else "fresh-test-token",
                "api_mode": "codex_responses"}

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", blocking_resolution)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length_async", AsyncMock(return_value=10000))
    monkeypatch.setattr("agent.context_references.preprocess_context_references_async",
                        AsyncMock(return_value=SimpleNamespace(blocked=False, expanded=False)))
    monkeypatch.setattr("agent.image_routing.decide_image_input_mode", lambda *a: "native")
    event = MessageEvent(text="hello @file" if caller == "context" else "describe",
                         message_type=MessageType.TEXT, source=source,
                         media_urls=["/test/image.png"] if caller == "image" else [],
                         media_types=["image/png"] if caller == "image" else [])

    async def heartbeat():
        while not entered.is_set():
            await asyncio.sleep(0)
        release.set()

    task = asyncio.create_task(heartbeat())
    try:
        if caller.startswith("hygiene"):
            # Execute the exact production resolution statements in isolation
            # from unrelated transcript/adapter/compaction setup. The main
            # preprocessing and background cases above run whole real callers.
            tree = ast.parse(textwrap.dedent(inspect.getsource(runner._handle_message_with_agent)))
            sites = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Tuple)
                     and ast.unparse(n.targets[0]) == "(_hyg_model, _hyg_runtime)"]
            sites.sort(key=lambda n: n.lineno)
            assert len(sites) == 2
            statement = sites[0 if caller == "hygiene_metadata" else 1]
            harness = ast.parse("async def resolve(self, source, session_key, _hyg_data):\n    pass\n")
            harness.body[0].body = [statement]
            namespace = {"asyncio": asyncio}
            exec(compile(ast.fix_missing_locations(harness), "<gateway hygiene callsite>", "exec"), namespace)
            await namespace["resolve"](runner, source, key, {})
        elif caller == "background":
            await runner._run_background_task_inner("hello", source, "task-test")
            adapter.send.assert_awaited_once()
            assert "no provider credentials" in adapter.send.call_args.args[1]
        else:
            result = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key=key)
            assert result == event.text
        await asyncio.wait_for(task, 2)
    finally:
        release.set()
        task.cancel()
    assert entered.is_set(), "Regression must reach actual managed provider resolution"
    assert not timed_out, "Managed auth blocked the event loop heartbeat"
    assert all(t != threading.get_ident() for t in threads)
