# BitGN ECOM Python Sample

Runnable Python sample for the `bitgn/ecom1-dev` benchmark.

Watch the leaderboard here - [https://bitgn.com/challenge/ecom](https://bitgn.com/challenge/ecom)

ECOM is an ecommerce operations runtime. It exposes a file-shaped workspace plus runtime tools such as `/bin/sql` over the `bitgn.vm.ecom` API.

This local copy can use `codex exec -m gpt-5.5` as the LLM backend, so it does
not require `OPENAI_API_KEY` when the current Codex subscription is logged in.
`BITGN_API_KEY` is still required for official ECOM benchmark runs and may be
placed in a local `.env` file.

## Setup

1. Put `BITGN_API_KEY=...` in `.env`, or export it in the shell.
2. Optionally export `BENCH_ID`, `BENCHMARK_ID`, `MODEL_ID`, or `LLM_PROVIDER`.
3. Run `make sync`
4. Run `make t01`, `make task TASKS="t01 t04"`, or `make run`

## Commands

- Run the full ECOM benchmark: `make run`
- Run t01: `make t01`
- Run a single task directly: `uv run python main.py t01`
- Run a subset of tasks directly: `uv run python main.py t01 t04`
- Run with local observability and task classification: `make monitor`
- Run selected tasks with observability: `make monitor-task TASKS="t01 t04"`
- Classify tasks without running the agent: `make classify`
- Show analytics summary: `make analytics`
- Regenerate markdown reports: `make report`
- List reusable agent skills: `make skills`
- Check that runtime rules/skills are not task-specific: `make generalization-check`
- Snapshot current runtime skills into observability evolution history: `make snapshot-skills`
- Show the current one-hypothesis evolution state: `make evolution`
- Install or update the local environment: `make sync`
- Run selected tasks via Make: `make task TASKS="t01 t04"`

Useful environment overrides:

- `BITGN_API_KEY` is required for official ECOM benchmark runs
- `BENCH_ID` or `BENCHMARK_ID` defaults to `bitgn/ecom1-dev`
- `MODEL_ID` defaults to `gpt-5.5`
- `LLM_PROVIDER` defaults to `codex`; set `LLM_PROVIDER=openai` to use `OPENAI_API_KEY`

## Local Observability

Monitor mode stores reproducible local analytics in `.bitgn_obs/`:

- `.bitgn_obs/obs.db` is the SQLite source of truth for runs, trials, events,
  LLM calls, tool calls, messages, classifications, failures, hypotheses,
  changes, and regressions.
- `.bitgn_obs/runs/<local_run_id>/events.jsonl` is the append-only raw event log.
- Trial directories store transcripts, full LLM prompts/raw outputs, tool
  payloads, score details, classifications, and failure analyses.
- `.bitgn_obs/reports/*.md` contains Codex-readable run, task, failure,
  hypothesis, and regression reports.
- `.bitgn_obs/workitems/` contains one-change fix workitems created by
  `analytics_cli.py new-fix`.

Useful direct commands:

```sh
uv run python main.py --monitor --classify-tasks t01
uv run python main.py --monitor --classify-tasks --classify-only
uv run python analytics_cli.py tasks
uv run python analytics_cli.py hypotheses
uv run python analytics_cli.py skills
uv run python analytics_cli.py check-generalization
uv run python analytics_cli.py snapshot-skills --title "baseline reusable skills"
uv run python analytics_cli.py evolution
uv run python analytics_cli.py new-fix --task t01 --hypothesis hyp_x --title "Verify exact rows before mutation"
uv run python analytics_cli.py finish-fix --change fix_x --status applied --summary "Added one verification step"
```

Secrets are recursively redacted before storage. Harness URLs are stored as
hashes, and environment values matching `KEY`, `TOKEN`, `SECRET`, or `PASSWORD`
are redacted.

## Reusable Skills And Evolution Guardrails

The runtime agent now loads generalized markdown skills from `skills/` based on
task family and risk. Skills describe reusable workflows only: catalog lookup,
inventory aggregation, data reconciliation, checkout/payment recovery,
discount/pricing policy, read-only investigations, authorization, completion,
and runtime tool-contract discipline.

Task-specific learning is intentionally blocked from runtime policy. The
`generalization_guard.py` scanner rejects task ids, commerce object ids, dates,
local host paths, concrete runtime evidence paths, and copied command/path facts
inside runtime rules or skills. Use:

```sh
uv run python analytics_cli.py check-generalization
```

The intended evolution loop is:

1. Collect monitored evidence.
2. Create or select one hypothesis.
3. Make one major generalized skill/rule/guard change.
4. Run `check-generalization`.
5. Run the target task, then the risk cluster, then a full monitor run.
6. Snapshot accepted skills with `snapshot-skills`.
