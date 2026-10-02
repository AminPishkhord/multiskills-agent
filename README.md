Multi-Skill Agent (Technical Task Prototype)

A LangGraph-based agent that takes a free-form prompt (English or Persian), narrows
down which skills could be relevant via **Skill Discovery**, routes it to 1-2 of them
via a **Router**, decides *how* to run them via a **Planner**, validates that plan,
and executes it generically via an **Orchestrator** that talks to a **Skill
Registry** - never to a specific skill by name. Four skills are registered today:
**Summarizer**, **Translator**, **Calculator**, **General Chat**. The architecture is
built so a 5th, 10th, or 30th skill can be added by registering it alone - no change
to the Router, Planner, Orchestrator, or LangGraph wiring. See `spec.md` for the
design document this implements.

Built as a prototype; not production-hardened (see [Known Limitations](#known-limitations-and-production-todos)).

## Table of contents
- [How to run](#how-to-run)
- [Architecture](#architecture)
- [Prompt engineering process](#prompt-engineering-process)
- [Multi-skill handling](#multi-skill-handling)
- [Fallback behavior](#fallback-behavior)
- [Persian-language support](#persian-language-support)
- [LLM provider](#llm-provider)
- [Evaluation](#evaluation)
- [Known limitations and production TODOs](#known-limitations-and-production-todos)
- [Assumptions](#assumptions)
- [AI usage disclosure](#ai-usage-disclosure)

## How to run

### Prerequisites
- Python 3.12
- A free API key from [OpenRouter](https://openrouter.ai/keys)

### Install
```bash
git clone <this-repo-url>
cd multiskills-agent
pip install -r requirements.txt
```

### Configure
```bash
cp .env.example .env
# then edit .env and paste your OPENROUTER_API_KEY
```

### Run
```bash
# interactive REPL (English or Persian, mix freely)
python -m src.main

# single prompt, non-interactive
python -m src.main --once "summarize this and translate it to English: ..."

# print discovery candidates, routed skills, reasoning, and the execution plan
python -m src.main --verbose --once "what's 12*7?"
```

### Run the tests
```bash
pytest
```
42 unit tests cover: the safe calculator tool, the structured-JSON parse/retry
helper, the `RoutingDecision`/`ExecutionPlan` Pydantic schemas, the Skill Registry,
Skill Discovery's TF-IDF candidate narrowing, Planner semantic validation (unknown
skills, dangling references, self-dependencies, cycles), and the generic Orchestrator
(single-step, parallel, and sequential-chained execution). All run fully offline with
no API key needed.

### Optional: Docker
```bash
docker build -t multiskills-agent .
docker run -it --env-file .env multiskills-agent
```

## Architecture

```
           ┌──────────────────┐
           │  Skill Registry  │   (summarizer, translator, calculator, general_chat)
           └────────┬─────────┘
                     │
                     ▼
User Prompt ───► Discovery ───► top-K candidate skill names
                     │
                     ▼
                  Router ───► 1-2 selected skills (validated subset of candidates)
                     │
                     ▼
                  Planner ───► ExecutionPlan (1-2 steps, sequential or parallel)
                     │
                     ▼ (semantic validation happens inside the Planner)
               Orchestrator ───► Registry.get(skill).run(input) for each step
                     │
                     ▼
                  Combine ───► Final Response
```

**LangGraph wiring** (`src/graph/graph.py`): `START -> discovery -> router -> planner
-> execute -> combine -> END`. Note what's *not* there: there is no per-skill node and
no conditional fan-out keyed on skill name. The graph's shape is fixed regardless of
how many skills are registered - dispatch to the actual skill implementation happens
generically inside the `execute` node, via `Orchestrator.execute(plan)` calling
`registry.get(skill_name).run(...)`. This is the concrete difference from a naive
"router -> [skill_a, skill_b, skill_c, skill_d] -> combine" graph (which is what an
earlier version of this project actually was - see "Why this changed" below):
scaling to 30 skills here means registering 30 skills, not adding 30 nodes and edges.

### Component-by-component reasoning

- **`src/registry/skill_registry.py` (Skill Registry)** - the single source of truth
  for "what skills exist and how to run them" (name, description, implementation).
  It does not decide anything (`Registry != Router`, `Registry != Discovery`, per
  spec.md section 4). `build_default_registry()` is the *only* place in the codebase
  that imports concrete skill classes by name - everything downstream only ever talks
  to `SkillSpec` objects by name string.

- **`src/skills/base.py` + `src/skills/*.py` (Skills)** - each skill is a
  `BaseSkill` subclass with a `name`, a `description` (used by both Discovery and the
  Router), and a `run(input_data: SkillInput) -> SkillOutput` method. Every skill
  shares the *same* input/output shape (`SkillInput(text=...)` / `SkillOutput(output=...)`)
  so the Orchestrator never needs to know a skill's internal fields - see
  [Assumptions](#assumptions) for why this uniform contract was chosen over each
  skill defining its own bespoke schema.

- **`src/discovery/skill_discovery.py` (Skill Discovery)** - embeds every registered
  skill's *description* once, embeds the incoming prompt, and returns the top-K most
  relevant skill names, so the Router's prompt size stays bounded regardless of
  registry size (spec.md sections 5 & 17). With only 4 skills registered today and
  `DISCOVERY_TOP_K=4`, this is a pass-through (`len(all) <= top_k` short-circuits to
  "return everything") - the same code scales down automatically as more skills are
  added. `general_chat` is always force-included as a candidate (fallback safety
  net) regardless of its similarity rank. See [Assumptions](#assumptions) for why
  TF-IDF + cosine similarity was used here instead of a neural embedding model.

- **`src/routing/router.py` (Router)** - builds a prompt listing *only* the
  candidates Discovery handed it, asks the LLM for structured JSON
  (`RoutingDecision`), and filters the result down to names that are actually in the
  candidate set (defense in depth against the model hallucinating a skill name).
  Returns 1-2 skills.

- **`src/routing/planner.py` (Planner + semantic Plan Validation)** - decides *how*
  the 1-2 selected skills should run. One skill = trivial one-step plan, no LLM call.
  Two skills = an LLM call decides sequential (`step_2.input_text = "$step_1.output"`)
  vs. parallel (both get the original prompt). `validate_plan()` then checks things
  Pydantic alone can't: do the referenced skills exist, do `depends_on` references
  point at real steps, is there a dependency cycle (generic DFS-based check, not
  hardcoded to 2 steps) - mirroring spec.md section 9's Structural-vs-Semantic split.

- **`src/orchestration/orchestrator.py` (Orchestrator)** - executes a *validated*
  plan generically: resolve each step's input text (literal, or substituted from a
  prior step's output), `registry.get(step.skill)`, call `.run(...)`. There is no
  `if skill == "calculator": ...` anywhere in this file (spec.md section 10).

- **`src/graph/graph.py` (LangGraph)** - controls the flow between the above
  components; it does not contain any skill-specific logic itself (spec.md section 15).

### Why this changed from an earlier version

An earlier version of this project used a LangGraph `Send`-based fan-out straight
from the Router to one hardcoded node per skill (`router -> [summarizer_node,
translator_node, calculator_node, general_chat_node] -> combine`). That worked for
exactly 4 skills, but adding skill #5 would have meant: writing a new node function,
adding it to the graph, adding an edge to `combine`, and extending the Router's fixed
`SkillName` enum - i.e. touching the graph itself every time, which directly
contradicts the extensibility principle this redesign is built around (spec.md
section 17: *"Adding a new Skill should not require modifying the core Agent
logic"*). The Registry/Discovery/Planner/Orchestrator layering above is what makes
"add a skill" mean "write a `BaseSkill` subclass and register it" and nothing else.

## Prompt engineering process

Each skill owns its own system prompt inline in its module (`src/skills/*.py`), plus
the Router's prompt (`src/routing/router.py`) and the Planner's prompt
(`src/routing/planner.py`). General process: draft a minimal instruction, run it
against a handful of the eval-set prompts (including Persian ones), look for failure
modes, then tighten the prompt against those specific failures.

### Documented example: the Router prompt

**Initial version (v1):**
```
You are a router. Decide which skills apply to the user's message: summarizer,
translator, calculator, general_chat. Return JSON with a "skills" list.
```

**Problems found when testing v1 against the eval set:**
1. On prompts like *"I have 3 cats and 2 dogs"* it sometimes returned
   `["calculator"]` - pattern-matching on digits rather than computational *intent*.
2. On Persian prompts it occasionally returned skill names as free-text Persian words
   instead of the exact candidate strings, which failed downstream filtering.
3. It sometimes returned 3+ skills for compound-sounding messages.
4. Ambiguous/general-knowledge prompts were inconsistently routed.

**Final version (v2)** - what changed and why (see `_build_system_prompt` in
`src/routing/router.py`):
- Added an explicit rule telling the model not to route to a specific skill just
  because a message "contains numbers or foreign words," and to prefer
  `general_chat` whenever unsure - fixed #1 and #4.
- The prompt now lists the *exact* candidate name strings (quoted) pulled live from
  Discovery's output, and the JSON schema example repeats those exact strings - fixed
  #2 regardless of input language.
- `RoutingDecision`'s Pydantic validator caps/dedupes to 2 server-side as defense in
  depth, and `decide_skills()` additionally filters out any name not in the candidate
  set - fixed #3 even on the rare occasions the prompt alone doesn't.
- Added bilingual (English + Persian) worked examples (`_CANONICAL_EXAMPLES`),
  included only for skills that are actually among the candidates.

The same iterate-on-observed-failures approach applied to the Summarizer prompt (had
to explicitly forbid adding opinions/info not in the source) and the Translator
prompt (had to explicitly forbid echoing the original text alongside the translation).

## Multi-skill handling

The Planner (`src/routing/planner.py`), not the LangGraph layer, decides how 2
selected skills run:
- **Sequential**: if the Planner's LLM call judges that skill B should operate on
  skill A's output (e.g. "summarize this, then translate it"), it emits a plan where
  step 2's `input_text` is `"$step_1.output"` and `depends_on: ["step_1"]`. The
  Orchestrator resolves that reference to the actual summary text before calling the
  translator - this was verified with a mocked-LLM integration test showing the
  translator skill receiving the *summary*, not the original long text.
- **Parallel/independent**: if the two skills answer unrelated parts of one message
  (e.g. "hi! also what's 12*7?"), both steps get the original prompt text and no
  dependency - the Orchestrator's "run whatever's ready" loop executes them in the
  same pass (sequentially in this prototype, not threaded - see
  [Known Limitations](#known-limitations-and-production-todos)).
- **Failure handling**: if the Planner's LLM call or `validate_plan()` fails for any
  reason, `decide_plan()` falls back to a deterministic parallel plan (both skills on
  the original prompt) rather than crashing. This can only lose the
  "chain-through-the-summary" optimization for that one request, never corrupt the
  response.
- **Combining outputs**: `combine_node` in `src/graph/graph.py` formats the 1-2
  resulting outputs with localized section headers ("Summary"/"Translation" vs.
  "خلاصه"/"ترجمه") when there are 2, or returns the single output as-is when there's 1.

## Fallback behavior

Multiple layers, all converging on **General Chat**:
1. **Router prompt**: explicitly instructed to prefer `general_chat` (when it's among
   the candidates) whenever a message doesn't clearly match a specific skill, rather
   than force-fitting the "closest" one.
2. **Discovery**: `general_chat` is *always* included in the candidate set handed to
   the Router, regardless of its TF-IDF similarity score, since it's the system's
   universal fallback and must stay reachable even on a low-scoring prompt.
3. **Router code**: if the LLM call fails entirely (bad JSON after retries, API
   error) or returns an empty/filtered-to-empty skill list, `router_node` in
   `src/graph/graph.py` hard-codes `skills = ["general_chat"]` (or, if `general_chat`
   somehow isn't registered, the first available candidate) with a reasoning string
   explaining the fallback - this guarantees the graph always has something runnable.

## Persian-language support

- **Routing**: the Router prompt explicitly states intent-based routing must work
  regardless of language, with Persian worked examples included.
- **Summarizer / Translator / General Chat**: each prompt instructs the model to
  detect the input language and respond in that same language (Translator defaults
  Persian<->English when no target language is stated - see
  [Assumptions](#assumptions)).
- **Calculator**: its expression-extraction prompt explicitly handles Persian number
  words and Persian/Arabic-indic digits, and the final response is templated in
  Persian when the input was Persian (simple Unicode-range check, `\u0600-\u06FF`, no
  extra LLM call - avoids any risk of an LLM call altering the computed number).
- **Discovery caveat**: Skill Discovery's TF-IDF similarity is computed over English
  skill *descriptions*. Today, with `top_k == len(registry)`, Discovery is a
  pass-through and this never matters. If the registry grows past `top_k`, a Persian
  query would be compared (lexically) against English descriptions, which TF-IDF
  handles poorly across languages - flagged explicitly in
  [Known Limitations](#known-limitations-and-production-todos) as something a real
  embedding model (most of which are far more language-robust) would fix.

## LLM provider

**Provider:** [OpenRouter](https://openrouter.ai) (free tier).
**Model:** `qwen/qwen3.8-27b:free` - OpenAI's open-weight gpt-oss-20b, **21 billion
parameters** total (Mixture-of-Experts, 3.6B active per forward pass; still ≤ 35B as
required), free tier (`:free` suffix, $0/token).

⚠️ This project's original default model, `qwen/qwen-2.5-32b-instruct:free`, was
retired by OpenRouter partway through development (a live run returned `400: ... is
not a valid model ID`) - a real instance of the free-catalog volatility worth
planning for, not a hypothetical. If `qwen/qwen3.8-27b:free` is also gone by the
time you read this, check
[openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0) for a
current ≤35B free model and set `OPENROUTER_MODEL` in `.env` - no code changes
needed. Free-tier shared-pool rate limits (`429`) are also expected occasionally;
retry after a short wait, switch models, or attach your own provider key via
[openrouter.ai/settings/integrations](https://openrouter.ai/settings/integrations).

## Evaluation

`eval/eval_set.json` has 15 prompts spanning clear single-skill cases (English +
Persian), multi-skill cases, and ambiguous/no-clear-skill cases. Run:

```bash
python -m eval.run_eval
```

This runs Discovery + Router (not the full Planner/Orchestrator pipeline, to keep
scoring fast/cheap) on each prompt and writes `eval/eval_report.md` with per-prompt
predicted-vs-expected skills, exact-match accuracy, and mean-Jaccard (partial credit
for multi-skill cases).

> **Note on this submission:** `eval/run_eval.py` is fully implemented and ready to
> run; the actual accuracy numbers should be (re)generated with a real
> `OPENROUTER_API_KEY` and reviewed before this is treated as a finished evaluation
> artifact, rather than filled in with fabricated numbers here.

## Known limitations and production TODOs

- **No conversation memory / no persistence / no auth / no web API** - explicitly out
  of scope for this task.
- **Skill Discovery uses TF-IDF, not a real embedding model** - a deliberate
  free/offline/dependency-light choice for the prototype (see
  [Assumptions](#assumptions)), but it matches on lexical overlap, not deep semantic
  similarity, and degrades across languages (a Persian query vs. English skill
  descriptions). Irrelevant today since `top_k == len(registry)` makes Discovery a
  pass-through; would need to be swapped for a real embedding API/model (the
  `EmbeddingIndex` interface in `src/discovery/skill_discovery.py` exists specifically
  so that's a one-class change) before scaling the registry meaningfully past ~10
  skills.
- **Orchestrator executes "ready" steps sequentially, not concurrently** - correct
  for parallel *plans* (both skills still run independently and don't see each
  other's output), but not literally parallel in wall-clock time. A production
  version would run same-wave steps with `asyncio.gather` or a thread pool.
- **Planner's sequential/parallel decision is itself an LLM call** - it can be wrong
  on an ambiguous 2-skill prompt; the deterministic parallel-fallback-on-failure
  only triggers on outright failures (bad JSON, invalid plan), not on a
  successfully-parsed-but-wrong decision.
- **Calculator scope** - arithmetic, percentages, roots, common functions, simple
  word problems; not symbolic algebra or multi-step reasoning problems.
- **Router/Planner accuracy depends on the free model's instruction-following** -
  smaller free models are noisier at structured output than larger ones; the
  JSON-parse retry loop mitigates but doesn't eliminate this.
- **No rate-limit/backoff handling beyond the JSON-parsing retry loop** - a `429`
  from the shared free-tier pool surfaces as a skill's error message (caught and
  shown cleanly, per `BaseSkill`'s contract) rather than being automatically retried.
- **No streaming output** in the CLI.

## Assumptions

Where spec.md left an implementation mechanism unspecified, the following choices
were made (and are called out again at the top of `spec.md` itself):
- **Discovery's embedding backend**: scikit-learn TF-IDF + cosine similarity instead
  of a neural embedding model/API. This keeps the prototype free, fully offline, and
  trivially testable (no network call, no model download), at the cost of only
  lexical (not deep semantic) and only mono-lingual-robust matching - acceptable
  today since 4 skills means Discovery is a pass-through regardless; flagged as a
  swap-out point for production (see [Known Limitations](#known-limitations-and-production-todos)).
- **Uniform Skill I/O contract**: every skill takes `SkillInput(text=...)` and
  returns `SkillOutput(output=...)`, rather than each skill defining bespoke fields
  (e.g. Calculator taking `expression` directly). This is what lets the Orchestrator
  and Planner stay fully generic with zero per-skill knowledge; each skill is still
  free to do whatever it wants internally with that `text` (e.g. Calculator extracts
  an expression from it internally).
- **Planner skips its LLM call for single-skill plans**: a trivial one-step plan is
  built directly, since there's no ordering decision to make - saves a request and a
  failure point for the common case.
- **Planner's failure-fallback is a parallel (not sequential) plan**: if the
  Planner's LLM call or semantic validation fails, defaulting to "run both skills
  independently on the original prompt" is always safe (never silently
  misinterprets a request), even though it loses the chaining optimization for that
  one request.
- **Translator's default target language** when none is stated: Persian<->English,
  whichever the source text *isn't*, given this task's explicit bilingual focus.
- **Fallback target**: General Chat, forced into every Discovery candidate set,
  rather than a dedicated "please clarify" dead-end node.
- **"Up to 2 simultaneous skills"**: enforced both in the Router's prompt and
  server-side (`RoutingDecision`'s validator caps/dedupes to 2 regardless of what the
  model returns), and again structurally in `ExecutionPlan` (`max_length=2` on steps).
- Docker support included as the brief's optional bonus, kept minimal.

## AI usage disclosure

An AI coding assistant was used throughout this project's design and implementation,
including a significant architecture revision based on a technical design document
provided by the project owner. See [`spec.md`](./spec.md) for that document verbatim
plus a summary of implementation decisions made where it didn't specify a mechanism.
