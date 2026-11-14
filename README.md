# MCPTox Gateway

An experiment on prompt injection in MCP tool descriptions. Four actor models respond to the same subset of the [MCPTox benchmark](https://github.com/zhiqiangwang4/MCPTox-Benchmark). A separate Claude Sonnet judge classifies each saved response, and we compare three paths:

- **A:** no reviewer.
- **B:** a reviewer checks the response.
- **C:** the reviewer also receives tool-description comparison evidence.

The main run uses 100 attack cases and 20 controls per actor. Original benchmark files are unchanged; some control contexts contain documented project-authored changes. The experiment evaluates model-generated text and does not execute proposed tools.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --locked
uv run --locked pytest -q
```

## View the results

Open [notebooks/experiment.ipynb](notebooks/experiment.ipynb) for the results, call coverage, case examples, and conclusions. Saved outputs are included. To rerun the notebook, the corresponding local run evidence must be available.

The [four-actor summary](results/four-actor-comparison-2026-11-13-v1/summary.json) contains the machine-readable results. The judge assessed fewer attacker-goal responses passing B and C than A for all four actors in these runs. These are judgments of proposed text, not executed attacks.

### Attack cases

Judge-assessed attacker-goal responses passed through each path, out of jointly classifiable attack cases:

| Actor | A: no review | B: reviewer | C: reviewer + evidence |
| --- | ---: | ---: | ---: |
| GPT-5.4 Mini | 24/100 | 2/100 | 4/100 |
| Qwen3.5 397B | 15/98 | 1/98 | 2/98 |
| MiniMax M3.1 Flash | 50/100 | 3/100 | 4/100 |
| Nemotron 3 Ultra | 18/100 | 1/100 | 2/100 |

### Control cases

Judge-assessed responses aligned with the request and passed through each path, out of 18 classifiable controls:

| Actor | A | B | C |
| --- | ---: | ---: | ---: |
| GPT-5.4 Mini | 18/18 | 16/18 | 17/18 |
| Qwen3.5 397B | 18/18 | 17/18 | 16/18 |
| MiniMax M3.1 Flash | 18/18 | 16/18 | 17/18 |
| Nemotron 3 Ultra | 18/18 | 16/18 | 18/18 |

Results are exploratory: the judge has not been independently validated, and a documented label inconsistency needs further audit. Two Qwen attack cases were excluded; Nemotron's 100-case result includes selected retries; and MiniMax acted as both actor and reviewer in separate calls. Two ambiguous controls per actor were excluded from control alignment metrics.

## Reproduce

Replay the historical run offline:

```sh
uv run --locked python -m mcptox_gateway.main reproduce --mode replay
```

Replay requires the local manifests, raw responses, traces, and call ledger. These files are ignored by Git, so a fresh clone cannot replay the historical run without them.

A fresh experiment requires an OpenRouter key in the environment or an ignored `.env` file. It downloads and verifies the pinned source, runs the pilot, then proceeds to the main run if the pilot passes:

```sh
uv run --locked python -m mcptox_gateway.main reproduce --mode fresh --experiment-id my-run --download --collect-missing
```

Fresh runs make paid API calls. The workspace instructions set a **$10 cumulative project ceiling**; the current configuration still records a $20 ceiling and $19.50 stop. Resolve that mismatch before collecting new calls. Existing charges count toward the ceiling.

## Project structure

| Path | Purpose |
| --- | --- |
| [config/](config/) | Experiment settings and reproduction lock. |
| [data/](data/) | Source manifest and locally prepared data. |
| [prompts/](prompts/) | Actor and reviewer instructions. |
| [src/mcptox_gateway/](src/mcptox_gateway/) | Experiment and evaluation code. |
| [src/mcptox_gateway/pipelines/](src/mcptox_gateway/pipelines/) | Data preparation, collection, evaluation, and replay. |
| [notebooks/](notebooks/) | Results and analysis. |
| [results/](results/) | Run summaries and local evidence. |
| [tests/](tests/) | Offline tests. |
| [explanation.md](explanation.md) | Detailed project introduction. |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Study design and amendments. |
