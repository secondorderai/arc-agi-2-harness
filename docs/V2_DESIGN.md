# ARC-AGI-2 V2 sequential program synthesis

V2 uses OpenAI `gpt-5.6-luna` with `xhigh` reasoning to synthesize reusable
`solve(train, grid)` induction programs. It does not fine-tune Luna. The local training
products are a verified program bank, a cross-task compatibility matrix, and a lightweight
retrieval ranker.

Open the [interactive V2 architecture diagram](V2_ARCHITECTURE.html) for the complete training,
freezing, evaluation, and checkpoint-resume flow.

## Run and resume

Install the optional ranker dependencies and sign the experiment workspace into Codex. The login
is isolated under the run workspace, uses ChatGPT subscription access, and never needs an API key:

```bash
uv sync --extra dev --extra v2
uv run arc-agent v2-auth-login --workspace runs/v2-main
uv run arc-agent v2-auth-status --workspace runs/v2-main
uv run arc-agent v2-build-bank \
  --data data/ARC-AGI-2/data/training \
  --config configs/v2-luna-xhigh.yaml \
  --workspace runs/v2-main
```

If `codex` is not on the terminal `PATH`, V2 automatically discovers the executable bundled with
ChatGPT on macOS at `/Applications/ChatGPT.app/Contents/Resources/codex`. You can also install the
standalone CLI with OpenAI's [official installer](https://developers.openai.com/codex/cli), or set
`openai.codex_cli` in the configuration to an explicit executable path.

The command is idempotently resumable. It records a prepared request before submission, stores a
Codex thread checkpoint before starting the turn, assigns the request key as the persisted client
message ID, and ingests each completed turn exactly once. The workspace-scoped App Server daemon
can finish an accepted turn even if the harness exits, and a restart recovers it by thread and turn
ID instead of issuing a duplicate. If the ChatGPT Codex allowance is exhausted it exits with status
`paused_quota` and code 75. After the subscription window resets, continue the exact task and
refinement round with:

```bash
uv run arc-agent v2-build-bank --workspace runs/v2-main --resume
```

Inspect progress and recorded usage without taking the workspace lock:

```bash
uv run arc-agent v2-status --workspace runs/v2-main --phase training
```

After all 1,000 tasks, compatibility measurement and ranker fitting complete, the bank and
manifest are frozen. Run the label-blind public evaluation with:

```bash
uv run arc-agent v2-evaluate \
  --data data/ARC-AGI-2/data/evaluation \
  --workspace runs/v2-main
```

Evaluation is also quota-resumable. Test labels are removed before any program or prompt sees
the task. The complete submission is atomically frozen and hashed before labels are loaded by
the scorer.

## Subscription transport and isolation

The default `openai.auth_mode` is `chatgpt_subscription`. V2 embeds the officially supported Codex
App Server protocol and requires its `account/read` result to report a ChatGPT login. It also checks
that the signed-in workspace exposes exactly `gpt-5.6-luna` with `xhigh` reasoning. The turn omits a
priority/Pro service tier, uses the App Server JSON output schema, and continues refinement in the
same persisted Codex thread.

The subscription backend does not expose the API's background-response or
`max_output_tokens` controls. App Server turns and the local daemon replace background polling;
the configured token ladder remains recorded for experimental compatibility but is marked as not
enforced in subscription response metadata. Exact token usage comes from App Server usage events.
Dollar amounts are API-equivalent estimates only and are not claimed as incremental subscription
charges.

For evaluation isolation, the App Server runs from an empty workspace-local directory with shell,
unified execution, web search, image inspection, agents, skills installation, apps, plugins, and
MCP unavailable. A completed turn containing any external/tool item is rejected as a configuration
pause. Only the sanitized task prompt and retrieved program text are sent to Luna.

The original Responses API transport remains available for controlled compatibility runs by
setting `openai.auth_mode: api_key`; only that explicit mode reads `OPENAI_API_KEY`. Because the
authentication mode is part of the experiment hash, switching an existing workspace between API
and subscription access is rejected as a material configuration change.

## Acceptance policy

A training program must exactly solve the labelled source tests, every leave-one-demonstration-
out episode, eight D4 analogues, and four deterministic foreground-colour permutations. Static
checks reject imports, external access, task identifiers, literal source grids, input-to-literal
grid comparisons, and oversized literal collections. Programs run in a resource-limited isolated
interpreter.

There is no task, time, round, or spend cap. Transient rate limits and server failures retry with
backoff. Quota and configuration failures are durable pauses and never advance the task cursor.
