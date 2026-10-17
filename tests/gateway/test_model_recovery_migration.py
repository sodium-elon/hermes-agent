"""Upgrade safety and internal events for primary model recovery."""
from datetime import timedelta

import pytest

from gateway.platforms.base import MessageEvent
from tests.gateway.test_model_inactivity_recovery import store_factory
from tests.gateway.test_session_model_override_persistence import _make_runner, _make_source, OVERRIDE


def test_unknown_age_legacy_override_recovers_once(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)
    runner = _make_runner(store)
    event = MessageEvent(text="continue", source=_make_source())
    runner._record_session_user_instruction(event, entry.session_key)
    assert store.get_model_override(entry.session_key) is None
    assert entry.last_user_instruction_at == event.timestamp.timestamp()
    # A newly selected override is not erased on the very next request.
    store.set_model_override(entry.session_key, OVERRIDE)
    runner._record_session_user_instruction(event, entry.session_key)
    assert store.get_model_override(entry.session_key)["model"] == OVERRIDE["model"]


@pytest.mark.asyncio
async def test_first_model_command_starts_durable_instruction_clock(store_factory, tmp_path, monkeypatch):
    from tests.gateway.test_model_command_flat_string_config import (
        _make_runner as make_command_runner, _make_event, _setup_isolated_home,
    )
    _setup_isolated_home(tmp_path, monkeypatch, {"default": "primary", "provider": "openai-codex"})
    runner = make_command_runner()
    runner.session_store = store_factory()
    event = _make_event("/model gpt-5.5")
    await runner._handle_model_command(event)
    key = runner._session_key_for_source(event.source)
    assert key in runner.session_store._entries
    entry = runner.session_store._entries[key]
    assert entry.last_user_instruction_at == event.timestamp.timestamp()
    assert runner.session_store.get_model_override(key)["model"] == "gpt-5.5"
    resumed = MessageEvent(text="next", source=event.source, timestamp=event.timestamp + timedelta(hours=25))
    runner._record_session_user_instruction(resumed, key)
    assert runner.session_store.get_model_override(key) is None


def test_internal_events_do_not_advance_instruction_clock(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    runner = _make_runner(store)
    first = MessageEvent(text="hello", source=_make_source())
    runner._record_session_user_instruction(first, entry.session_key)
    store.set_model_override(entry.session_key, OVERRIDE)
    internal = MessageEvent(text="background completion", source=first.source,
                            timestamp=first.timestamp + timedelta(hours=25), internal=True)
    runner._record_session_user_instruction(internal, entry.session_key)
    assert entry.last_user_instruction_at == first.timestamp.timestamp()
    assert store.get_model_override(entry.session_key) is not None


def test_idle_cached_fallback_recovers_without_explicit_override(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    runner = _make_runner(store)
    first = MessageEvent(text="hello", source=_make_source())
    runner._record_session_user_instruction(first, entry.session_key)
    runner._agent_cache = {entry.session_key: (object(), "fallback-signature")}
    runner._agent_cache_lock = None
    resumed = MessageEvent(text="next", source=first.source,
                           timestamp=first.timestamp + timedelta(hours=25))
    runner._record_session_user_instruction(resumed, entry.session_key)
    assert entry.session_key not in runner._agent_cache


def test_sqlite_routing_clock_survives_restart(tmp_path, monkeypatch):
    import hermes_state
    from gateway.config import GatewayConfig, SessionResetPolicy
    from gateway.session import SessionStore
    real_db = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: real_db)
    config = GatewayConfig()
    config.default_reset_policy = SessionResetPolicy(mode="none")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    entry = store.get_or_create_session(_make_source())
    first = MessageEvent(text="hello", source=_make_source())
    runner = _make_runner(store)
    runner._record_session_user_instruction(first, entry.session_key)
    store.set_model_override(entry.session_key, OVERRIDE)
    reloaded = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    assert reloaded.get_or_create_session(first.source).last_user_instruction_at == first.timestamp.timestamp()
    restarted = _make_runner(reloaded)
    resumed = MessageEvent(text="next", source=first.source,
                           timestamp=first.timestamp + timedelta(hours=25))
    restarted._record_session_user_instruction(resumed, entry.session_key)
    assert reloaded.get_model_override(entry.session_key) is None
    real_db.close()
