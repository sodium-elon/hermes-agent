"""Managed auth must reach the cached consumer, not just runtime resolution."""
import sys
import threading
import types
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("active_fallback", [False, True])
async def test_cached_agent_consumes_replacement_pool_with_unchanged_token(monkeypatch, tmp_path, active_fallback):
    pools = [SimpleNamespace(provider="openai-codex") for _ in range(3)]
    current = {"provider": "openai-codex", "api_key": "same-test-token",
               "base_url": "https://chatgpt.com/backend-api/codex",
               "api_mode": "codex_responses", "credential_pool": pools[0]}
    consumed = []
    instances = []
    fallback_pool = SimpleNamespace(provider="openrouter")

    class Consumer:
        def __init__(self, **kwargs):
            instances.append(self)
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self._credential_pool = kwargs["credential_pool"]
            self.tools = []
            self.session_id = "test-session"
            self._cached_system_prompt = "frozen"

        def run_conversation(self, message, conversation_history=None, task_id=None):
            consumed.append((self.provider, self._credential_pool, self._cached_system_prompt))
            if active_fallback:
                self.provider = "openrouter"
                self.base_url = "https://openrouter.ai/api/v1"
                self._credential_pool = fallback_pool
            return {"final_response": "done", "messages": [], "api_calls": 1}

    module = types.ModuleType("run_agent")
    module.AIAgent = Consumer
    monkeypatch.setitem(sys.modules, "run_agent", module)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", lambda p: dict(current))
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = GatewayConfig()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {"test": {
        "model": "gpt-5.4", **current, "_managed_credentials": True,
    }}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm")

    for pool in pools:
        current["credential_pool"] = pool
        result = await runner._run_agent(message="hello", context_prompt="", history=[],
                                         source=source, session_id="test-session", session_key="test")
        assert result["final_response"] == "done"

    assert len(instances) == 1, "Pool identity must not bust the frozen prompt cache"
    expected = [pools[0], fallback_pool, fallback_pool] if active_fallback else pools
    assert len(consumed) == len(expected)
    assert all(row[1] is pool for row, pool in zip(consumed, expected)), "Cached consumer retained an old pool or crossed the fallback boundary"
    assert all(row[0] == pool.provider and row[2] == "frozen"
               for row, pool in zip(consumed, expected))
