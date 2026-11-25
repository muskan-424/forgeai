> **Status:** this is the original design plan. [README.md](README.md) describes what was actually built, including the changes real runs forced.

# ForgeAI - Architecture

ForgeAI takes a one-paragraph brief plus a folder of documents and autonomously produces a tested
agent definition (system prompt, retrieval config, tools, output contract), an auto-generated eval
set, an iteration log, and an honest quality report.

## Principles
1. Deterministic checks first, LLM second - every LLM judgement has a mechanical cross-check.
2. Typed contracts (pydantic) between stages; each stage's output is saved as JSON.
3. Targeted typed patches, never full regeneration.
4. Everything is logged into a self-contained `runs/<name>/` folder.
5. Unattended by default (`--auto`), with a hard time/token budget.
6. No domain-specific code: the factory must work on unseen briefs.

## Pipeline
```
brief + docs/
  -> [1 Ingest]        inventory + chunk index
  -> [2 Intake]        <=5 questions -> requirements.json/.md
  -> [3 Groundability] coverage matrix: supported / partial / unsupported
  -> [4 Generator]     agent_definition.yaml (prompt, retrieval, tools, output contract)
  -> [5 Eval generator] evals.json (dev + holdout split)
  -> loop <=3: [6 run agent] -> [7 judge] -> [8 critic classify] -> [9 optimizer patch]
  -> [10 Report]       definition + scorecards + changelog + honest failure analysis
```

## Components
| Module | Role |
|---|---|
| `forge/llm.py` | Ollama client (JSON-schema output, repair retry, disk cache, token/latency accounting) + Mock backend for tests |
| `forge/ingest.py` | PDF/MD/TXT/CSV/DOCX loaders, structure-aware chunking, document inventory |
| `forge/retrieval.py` | Own BM25 + optional embeddings (Ollama `nomic-embed-text`), hybrid fusion |
| `forge/tools/` | Registry: `csv_lookup`, `calculator`, `open_meteo`, `current_date`, each with a manifest + preconditions |
| `forge/agent_runtime.py` | Retrieve -> prompt -> optional tool action -> output-contract validation + citation check |
| `forge/intake.py` | Question drafting, filtering of doc-answerable questions, requirements doc |
| `forge/groundability.py` | Per-topic similarity floor + LLM entailment; both must agree for "supported" |
| `forge/generator.py` | Tool gating + justification, retrieval params from doc stats, system prompt |
| `forge/evalgen.py` | Verbatim-quote-verified cases: answerable, two-doc, out-of-scope, ambiguous |
| `forge/judge.py` | Deterministic checks + pointwise per-fact LLM judge, calibrated before each run |
| `forge/critic.py` / `optimizer.py` | Failure classification + typed patch ops, revert on regression |
| `forge/report.py` | Scorecards, changelog, Wilson CIs, residual failures with root causes |
| `chat.py` | Generic runner that loads any produced definition |

## Judge bias defences
- Position bias: pointwise scoring, never pairwise.
- Verbosity bias: per-fact yes/no rubric; deterministic fact matching first.
- Self-preference: judge model is a different family from the agent model.
- Instability: two passes with shuffled fact order; disagreement => "uncertain" => scored conservatively.
- Calibration: gold / fact-swapped / padded / empty answers scored before each run; accuracy reported.

## Critic categories -> fix operators
| Category | Signal | Fix |
|---|---|---|
| retrieval miss | expected quote not in top-k | tune k, chunk size, overlap, hybrid weight |
| hallucination | claim/citation unsupported, or answered an out-of-scope case | grounding/citation rules, lower temperature |
| wrong refusal | refused although answer was in context | patch refusal-threshold rule |
| prompt defect | context had answer, output wrong/malformed | patch the specific rule |

## Eval set
~30 cases: ~14 answerable, ~5 two-document, ~6 out-of-scope, ~5 ambiguous. Every answerable case
carries a supporting quote that must appear verbatim in the source, else it is discarded.
Split into dev (optimizer sees it) and holdout (reporting only) to expose overfitting.

## Default models (hardware: CPU only, ~7 GB RAM)
- Agent + generation: `qwen2.5:3b`
- Judge (different family): `llama3.2:3b`
- Embeddings: `nomic-embed-text`
All configurable in `forge/config.py` / env vars (`FORGE_AGENT_MODEL`, `FORGE_JUDGE_MODEL`, `FORGE_EMBED_MODEL`, `FORGE_BACKEND=ollama|mock`).

## Build order
1. Schemas, LLM wrapper, ingest, retrieval, tools, agent runtime, chat runner  (milestone 1)
2. Eval generator + judge (with calibration)
3. Intake, groundability guard, generator
4. Critic, optimizer, report
5. Two demo runs, then a third unseen dry run
6. Stretch: budget/early stop, human checkpoint, regression mode, prompt tournament, UI
