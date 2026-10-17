"""Model recovery uses durable real inbound activity, never transcript writes."""
from datetime import timedelta
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, SessionResetPolicy
from gateway.session import SessionStore
from tests.gateway.test_session_model_override_persistence import _make_source, _make_runner, OVERRIDE


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: None)

    def make():
        config = GatewayConfig()
        config.default_reset_policy = SessionResetPolicy(mode="none")
        return SessionStore(sessions_dir=tmp_path, config=config)
    return make


def test_user_instruction_clock_survives_restart_not_generic_activity(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    then = entry.created_at.timestamp()
    store.note_user_instruction(entry.session_key, then)
    with patch("gateway.session._now", return_value=entry.created_at + timedelta(hours=25)):
        store.get_or_create_session(_make_source())
    restored = store_factory().get_or_create_session(_make_source())
    assert restored.last_user_instruction_at == then


@pytest.mark.parametrize("hours, expires", [(23, False), (24, False), (24.0001, True)])
def test_recovery_boundary_clears_override_and_fallback_cache(store_factory, hours, expires):
    from gateway.platforms.base import MessageEvent
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    then = entry.created_at
    store.note_user_instruction(entry.session_key, then.timestamp())
    store.set_model_override(entry.session_key, OVERRIDE)
    runner = _make_runner(store)
    runner._session_model_overrides[entry.session_key] = dict(OVERRIDE)
    runner._running_agents = {}
    runner._agent_cache = {entry.session_key: (object(), "signature")}
    runner._agent_cache_lock = None
    runner._last_resolved_model = {entry.session_key: "fallback"}
    runner._pending_model_notes = {entry.session_key: "stale switch"}
    event = MessageEvent(text="continue", source=_make_source(), timestamp=then + timedelta(hours=hours))
    runner._record_session_user_instruction(event, entry.session_key)
    assert (store.get_model_override(entry.session_key) is None) is expires
    assert (entry.session_key not in runner._session_model_overrides) is expires
    assert (entry.session_key not in runner._agent_cache) is expires
    assert (entry.session_key not in runner._last_resolved_model) is expires
    assert (entry.session_key not in runner._pending_model_notes) is expires


def test_busy_recovery_is_deferred_and_survives_restart(store_factory):
    from gateway.platforms.base import MessageEvent
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    then = entry.created_at
    store.note_user_instruction(entry.session_key, then.timestamp())
    store.set_model_override(entry.session_key, OVERRIDE)
    runner = _make_runner(store)
    runner._running_agents = {entry.session_key: object()}
    event = MessageEvent(text="next", source=_make_source(), timestamp=then + timedelta(hours=25))
    runner._record_session_user_instruction(event, entry.session_key)
    assert store.get_model_override(entry.session_key) is not None
    # The user clock advanced while busy; expiry must still survive a crash.
    restarted = _make_runner(store_factory())
    restarted._running_agents = {}
    restarted._record_session_user_instruction(event, entry.session_key)
    assert restarted.session_store.get_model_override(entry.session_key) is None


@pytest.mark.asyncio
async def test_real_dispatch_recovers_before_model_command(store_factory, monkeypatch):
    from tests.gateway.test_session_race_guard import _make_runner as make_dispatch_runner
    from gateway.platforms.base import MessageEvent
    from unittest.mock import AsyncMock
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    then = entry.created_at
    store.note_user_instruction(entry.session_key, then.timestamp())
    store.set_model_override(entry.session_key, OVERRIDE)
    runner = make_dispatch_runner()
    runner.session_store = store
    runner._session_model_overrides = {entry.session_key: dict(OVERRIDE)}
    runner._handle_model_command = AsyncMock(return_value="picker")
    event = MessageEvent(text="/model", source=_make_source(), timestamp=then + timedelta(hours=25))
    assert await runner._handle_message(event) == "picker"
    assert store.get_model_override(entry.session_key) is None
    assert runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_pending_recovery_consumed_after_slot_claim_before_turn_setup(store_factory):
    from tests.gateway.test_session_race_guard import _make_runner as make_dispatch_runner
    from gateway.platforms.base import MessageEvent
    from gateway.run import _AGENT_PENDING_SENTINEL
    from unittest.mock import AsyncMock
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    then = entry.created_at
    store.note_user_instruction(entry.session_key, then.timestamp())
    store.set_model_override(entry.session_key, OVERRIDE)
    event = MessageEvent(text="continue", source=_make_source(), timestamp=then + timedelta(hours=25))
    runner = make_dispatch_runner()
    runner.session_store = store
    runner._session_model_overrides = {entry.session_key: dict(OVERRIDE)}
    runner._running_agents[entry.session_key] = _AGENT_PENDING_SENTINEL
    runner._record_session_user_instruction(event, entry.session_key)
    assert store.get_model_override(entry.session_key) is not None
    entry.is_fresh_reset = True  # stop after session resolution, before any LLM work
    runner.hooks.emit = AsyncMock(side_effect=RuntimeError("boundary reached"))
    with pytest.raises(RuntimeError, match="boundary reached"):
        await runner._handle_message_with_agent(event, event.source, entry.session_key, 1)
    assert store.get_model_override(entry.session_key) is None
    assert entry.last_user_instruction_at == event.timestamp.timestamp()
    assert runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_help_refreshes_only_recovered_topic_before_early_return(store_factory):
    from dataclasses import replace
    from unittest.mock import AsyncMock, Mock
    from gateway.platforms.base import MessageEvent
    from tests.gateway.test_session_race_guard import _make_runner as make_dispatch_runner

    store = store_factory()
    raw_source = replace(_make_source(), thread_id="old-topic")
    active_source = replace(raw_source, thread_id="active-topic")
    raw_entry = store.get_or_create_session(raw_source)
    active_entry = store.get_or_create_session(active_source)
    then = active_entry.created_at
    for entry in (raw_entry, active_entry):
        store.note_user_instruction(entry.session_key, then.timestamp())
        store.set_model_override(entry.session_key, OVERRIDE)
    persisted_override = store.get_model_override(active_entry.session_key)
    runner = make_dispatch_runner()
    runner.session_store = store
    runner._session_model_overrides = {
        entry.session_key: dict(OVERRIDE) for entry in (raw_entry, active_entry)
    }
    runner._recover_telegram_topic_thread_id = Mock(return_value="active-topic")
    runner._handle_help_command = AsyncMock(return_value="help")

    help_event = MessageEvent(
        text="/help", source=raw_source, timestamp=then + timedelta(hours=23)
    )
    assert await runner._handle_message(help_event) == "help"
    runner._handle_help_command.assert_awaited_once_with(help_event)
    assert active_entry.last_user_instruction_at == help_event.timestamp.timestamp()
    assert raw_entry.last_user_instruction_at == then.timestamp()
    assert raw_source.thread_id == "old-topic"

    next_event = MessageEvent(
        text="/help", source=raw_source, timestamp=then + timedelta(hours=25)
    )
    assert await runner._handle_message(next_event) == "help"
    assert active_entry.last_user_instruction_at == next_event.timestamp.timestamp()
    assert raw_entry.last_user_instruction_at == then.timestamp()
    for entry in (raw_entry, active_entry):
        assert store.get_model_override(entry.session_key) == persisted_override
        assert runner._session_model_overrides[entry.session_key] == OVERRIDE
        assert not entry.model_recovery_pending


def test_same_persisted_override_selection_supersedes_pending_recovery(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    then = entry.created_at
    store.note_user_instruction(entry.session_key, then.timestamp())
    store.set_model_override(entry.session_key, OVERRIDE)
    persisted_override = store.get_model_override(entry.session_key)
    assert store.note_user_instruction(
        entry.session_key, (then + timedelta(hours=25)).timestamp()
    )
    assert entry.model_recovery_pending

    # Selecting exactly the same persisted value must still cancel recovery.
    store.set_model_override(entry.session_key, persisted_override)
    restarted = store_factory()
    restored = restarted.get_or_create_session(_make_source())
    assert not restored.model_recovery_pending
    assert not restarted.consume_model_recovery(entry.session_key)
    assert restarted.get_model_override(entry.session_key) == persisted_override
