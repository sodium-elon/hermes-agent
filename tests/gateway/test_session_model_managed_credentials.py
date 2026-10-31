"""Live /model selections pin routes without pinning managed auth snapshots."""

import pytest
import yaml
from types import SimpleNamespace

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, SessionStore


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_endpoint", [False, True])
@pytest.mark.parametrize("picker", [False, True])
async def test_model_switch_refreshes_only_managed_credentials(tmp_path, monkeypatch, explicit_endpoint, picker):
    route = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    config: dict = {"model": {"default": "gpt-5.4", **route}}
    target = "openai-codex"
    if explicit_endpoint:
        target = "private"
        route = {"provider": target, "base_url": "https://private.example/v1", "api_mode": "chat_completions"}
        config["providers"] = {target: {"base_url": route["base_url"], "api_key": "old-test-token", "models": ["gpt-5.4"]}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: {"accepted": True})
    old_pool, fresh_pool = object(), object()
    current = {**route, "api_key": "old-test-token", "credential_pool": old_pool}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: dict(current))

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = GatewayConfig()
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    event = MessageEvent(
        text="/model --session" if picker else f"/model gpt-5.4 --provider {target} --session",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm"),
    )
    if picker:
        class Picker:
            async def send_model_picker(self, **kwargs):
                self.select = kwargs["on_model_selected"]
                return SimpleNamespace(success=True)

        adapter = Picker()
        runner.adapters = {Platform.TELEGRAM: adapter}
        monkeypatch.setattr("hermes_cli.model_switch.list_picker_providers", lambda **kw: [{"slug": target}])
        await runner._handle_model_command(event)
        response = await adapter.select(event.source.chat_id, "gpt-5.4", target)
    else:
        response = await runner._handle_model_command(event)
    key = runner._session_key_for_source(event.source)
    assert key in runner._session_model_overrides, response
    assert runner._session_model_overrides[key]["api_key"] == "old-test-token"
    current.update(api_key="fresh-test-token", credential_pool=fresh_pool)

    model, runtime = runner._resolve_session_agent_runtime(session_key=key)

    if explicit_endpoint:
        assert runtime["api_key"] == "old-test-token"
    else:
        assert runtime["api_key"] == current["api_key"]
        assert runtime["credential_pool"] is fresh_pool
    assert model == "gpt-5.4"
    for field, value in route.items():
        assert runtime[field] == value


@pytest.mark.parametrize("provider", ["custom:private", "openai-codex"])
def test_unmarked_explicit_key_and_pool_are_not_replaced(monkeypatch, provider):
    pool = object()
    override = {
        "model": "explicit-model", "provider": provider,
        "base_url": "https://private.example/v1", "api_key": "explicit-test-key",
        "credential_pool": pool,
    }
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {"test": override}

    def unexpected_resolution(*args, **kwargs):
        pytest.fail("Explicit credentials must not be replaced by provider resolution")

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", unexpected_resolution)
    model, runtime = runner._resolve_session_agent_runtime(session_key="test", user_config={})
    assert model == override["model"]
    for field in ("provider", "base_url", "api_key", "credential_pool"):
        assert runtime[field] == override[field]


@pytest.mark.asyncio
async def test_same_provider_switch_auth_failure_is_noop_then_recovers(tmp_path, monkeypatch):
    route = {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex",
             "api_mode": "codex_responses"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": {"default": "gpt-5.4", **route}}))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: {"accepted": True})
    monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", lambda *a: None)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = GatewayConfig()
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    event = MessageEvent(text="/model gpt-5.5 --session", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm"))
    runner.session_store.get_or_create_session(event.source)
    key = runner._session_key_for_source(event.source)
    runner._record_session_user_instruction(event, key)
    original = {"model": "gpt-5.4", **route, "api_key": "old-test-token", "_managed_credentials": True}
    runner._session_model_overrides[key] = dict(original)

    def unavailable(**kwargs):
        raise RuntimeError("transient auth-store failure")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", unavailable)
    response = await runner._handle_model_command(event)
    assert runner._session_model_overrides[key] == original, response
    assert "transient auth-store failure" in response

    fresh_pool = object()
    current = {**route, "api_key": "fresh-test-token", "credential_pool": fresh_pool}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: dict(current))
    model, runtime = runner._resolve_session_agent_runtime(session_key=key)
    assert model == "gpt-5.4"
    assert runtime["api_key"] == "fresh-test-token"
    assert runtime["credential_pool"] is fresh_pool
    await runner._handle_model_command(event)
    assert runner._session_model_overrides[key]["_managed_credentials"] is True
    assert runner._session_model_overrides[key]["model"] == "gpt-5.5"


def test_managed_resolution_failure_does_not_reuse_stale_credentials(monkeypatch):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {"test": {
        "model": "managed-model", "provider": "openai-codex",
        "api_key": "old-test-token", "credential_pool": object(),
        "_managed_credentials": True,
    }}

    def unavailable(provider):
        raise RuntimeError("Managed credentials unavailable")

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", unavailable)
    with pytest.raises(RuntimeError, match="Managed credentials unavailable"):
        runner._resolve_session_agent_runtime(session_key="test", user_config={})
