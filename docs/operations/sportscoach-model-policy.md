# Sportscoach model policy

This deployment uses the Sodium Hermes fork, pinned to the installed Hermes
baseline rather than an unrelated upstream upgrade. The fork's `main` is the
release branch. `upstream-main` retains the upstream snapshot from fork creation.
No credentials or profile session databases belong in Git.

## Required profile configuration

```yaml
model:
  default: gpt-6.1-sol
  provider: openai-codex
  persist_switch_by_default: false
fallback_providers:
  - provider: openrouter
    model: deepseek/deepseek-v4-pro
```

A temporary model selection or automatic fallback must not overwrite the primary.
Only an explicitly permanent `--global` selection changes the configured primary.
Ordinary gateway selections are session-scoped (the standalone CLI's existing
persistence preference is unchanged). On the first user instruction after more
than 24 hours of user inactivity in that conversation, temporary selection and
cached fallback state must no longer prevent a primary-first attempt. Exactly
24 hours is not more than 24 hours. Legacy routing entries without a user-only
clock return to primary once on their first instruction after upgrade; their age
cannot safely be inferred from generic routing or assistant timestamps. Recovery
happens before a new turn, never by interrupting an active turn. Provider
authentication/quota failures may still
activate the configured fallback; this policy does not bypass provider limits.

## Scoped activation

1. Verify regression tests and an independent review before publishing.
2. Publish the reviewed change to this fork's main and synchronize the canonical
   source checkout before installation.
3. Back up the profile config and session routing index. Apply only the model
   settings above; preserve all other configuration, secrets and conversation
   history. Inspect saved session overrides as well as config.yaml.
4. Preserve unrelated installed core changes. Compare hashes for deployed files
   against the release commit; do not claim the entire install is clean when
   unrelated local edits remain.
5. Confirm Sportscoach has no active task before restarting only
   `hermes-gateway-sportscoach.service`.
6. Verify authenticated API readiness, messaging platform connections, release
   file hashes and a harmless text-only API turn. Read actual provider/model
   logs: health alone does not prove primary selection.
7. Prove the inactivity boundary and restart/persistence behavior using isolated
   session stores and controlled clocks, not by aging a real conversation or
   replaying an old command. Do not run unsolicited speaker tests.

A failed live OpenAI request is not evidence of a permanent fallback change.
Report the attempted primary and any actual fallback/error separately.
