# tau-bench re-analysis: what does each agent run cost?

tau-bench published the full transcripts of every run of two agents, but not what each
run cost. This experiment rebuilds the agent's token usage for every model call in
those transcripts and prices it, so we can study how cost varies when the **same task**
is run several times.

## Data

| | |
|---|---|
| Source | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), MIT license |
| Commit | `59a200c6d575d595120f1cb70fea53cef0632f6b` |
| Files | `historical_trajectories/{gpt-4o,sonnet-35-new}-{airline,retail}.json` |
| Runs | 1,980: GPT-4o 50 airline + 115 retail tasks × 4 trials; Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`) the same tasks × 8 trials |

Each run has `task_id`, `trial`, `reward` (1.0 = success, 0.0 = failure), `info` and
`traj` (the full message list). The downloaded data is not committed.

## Reproduce

```bash
uv sync
uv run python experiments/taubench-reanalysis/download.py    # fetches the pinned commit into data/
uv run python experiments/taubench-reanalysis/build_runs.py  # writes results/runs.csv, prints checks
```

`download.py` records the commit in `data/SOURCE.json`. No tau-bench code is run: tool
definitions are read from its source files with Python's `ast` module.

## Output: `results/runs.csv`

One row per run.

| Column | Meaning |
|---|---|
| `model`, `domain`, `task_id`, `trial` | Which run |
| `success` | 1 if reward = 1.0, else 0 |
| `n_agent_calls` | Model calls by the agent (one per assistant message) |
| `n_tool_calls` | Tool calls by the agent |
| `input_tokens`, `output_tokens` | Estimated agent tokens, summed over all calls in the run |
| `cost_usd_main` | Agent cost at the main price assumption |
| `cost_usd_alt` | GPT-4o only: cost at the alternative snapshot's price |
| `user_cost_usd` | Simulated customer's cost as recorded by tau-bench. GPT-4o only (see below) |

## Method

Each assistant message in a transcript is one model call. A chat model has no memory
between calls, so every call re-sends the whole conversation so far. For assistant
message *i*:

- **input** = tool definitions + hidden tool prompt + every message before *i*
  (each with a per-message formatting overhead) + reply priming
- **output** = assistant message *i* (its text, plus its tool call for GPT-4o)

Tokens are counted with `tiktoken`'s `o200k_base`, GPT-4o's tokenizer. Code:
`src/burnscope/tokens.py` (token rules), `src/burnscope/importers/taubench.py` (reading
the files), `src/burnscope/pricing.py` (prices and cost).

### Constants and where they come from

| Constant | Value | Source |
|---|---|---|
| Per-message overhead | 3 tokens (+1 if the message has a `name`) | OpenAI cookbook *How to count tokens with tiktoken*, commit `db31449` (2024-10-08) |
| Reply priming | 3 tokens per call | Same cookbook: "every reply is primed with `<\|start\|>assistant<\|message\|>`" |
| GPT-4o tool definitions | Cookbook formula: 7 per tool, 3 + 3 per parameter, 12 at the end, plus name/description text | Same cookbook, `num_tokens_for_tools` |
| Claude tool definitions | Token count of the tool list as JSON text | Anthropic docs: tools are shown to the model "in JSONSchema format" |
| Claude hidden tool prompt | 346 tokens per call | Archived Anthropic tool-use docs ([2024-11-15](https://web.archive.org/web/20241115021342/https://docs.anthropic.com/en/docs/build-with-claude/tool-use)), Claude 3.5 Sonnet (Oct), `tool_choice` auto |

Resulting tool definition tokens per call: GPT-4o 1,120 (airline) and 1,875 (retail);
Claude 2,437 (airline) and 2,854 (retail), plus the 346-token hidden prompt.

### Prices (US $ per million tokens, standard API, late 2024)

| Model | Input | Output | Role | Source |
|---|---|---|---|---|
| `gpt-4o-2024-05-13` | 5.00 | 15.00 | GPT-4o main assumption (runs come from the 2024 paper) | [OpenAI pricing, archived 2024-11-15](https://web.archive.org/web/20241115053932/https://openai.com/api/pricing/) |
| `gpt-4o-2024-08-06` | 2.50 | 10.00 | GPT-4o sensitivity check | same |
| `claude-3-5-sonnet-20241022` | 3.00 | 15.00 | Claude | [Anthropic models page, archived 2024-11-13](https://web.archive.org/web/20241113202656/https://docs.anthropic.com/en/docs/about-claude/models) |

These versions are no longer on the live pricing pages, so archived official pages are
used. All three prices are verified; see `pricing.yaml`.

## Assumptions

1. **No prompt caching.** All input is priced as uncached, because the transcripts don't
   say whether caching happened. With caching, real costs could be lower.
2. **The GPT-4o snapshot is unknown.** The 2024-05-13 price is the main assumption;
   2024-08-06 (half the input price) is reported alongside it.
3. **Every call sends the system prompt, all earlier messages and the tool definitions**,
   as tau-bench's `tool_calling_agent.py` does. The GPT-4o transcripts match that code's
   format. The Claude transcripts do not (see limitations), so for Claude this is an
   assumption.

## Limitations

1. **Tool definition counting is the largest uncertainty.** Each model uses its
   provider's documented format, but neither is exact: the OpenAI formula ignores
   nested parameter fields, and Anthropic does not specify the exact JSON layout.
   Swapping the two methods changes median cost per run by:

   | | GPT-4o | Claude |
   |---|---|---|
   | airline | +38% | −21% |
   | retail | +23% | −15% |

   Within-model comparisons (same task across trials) are barely affected, because
   every call of a model gets the same fixed amount. **Cross-model comparisons are
   sensitive to this.**
2. **Claude's tokenizer is not public.** `o200k_base` is exact for GPT-4o text and only a
   proxy for Claude.
3. **Per-message overhead for Claude** uses OpenAI's published values as a proxy;
   Anthropic does not publish them.
4. **Claude transcripts have a different format** from tau-bench's current agent code:
   each tool call appears twice, as text inside the message content and again as a
   structured `tool_calls` entry. Only the text is counted, so the call is not counted
   twice.
5. **Extra parallel tool calls were discarded by tau-bench.** The agent code keeps only
   the first tool call of each message (`tool_calls[:1]`), so output tokens for any
   additional calls are missing. Output may be slightly underestimated.
6. **`user_cost` is a placeholder in the Claude files.** In every Claude run that has
   a value, it equals exactly 0.1 × the number of user messages, so it is not dollars
   and `user_cost_usd` is left empty. The GPT-4o values look like real dollars
   (median $0.0023 per run). 7 GPT-4o runs have no recorded value.
7. **Runs are capped at 30 agent calls** (`max_num_steps=30`). 19 runs hit the cap and
   none of them succeeded.

## Sanity checks

Output of `build_runs.py` (all checks pass):

```
model                       domain    runs tasks trials  success  paper calls p50  max  cost p50 $  check
---------------------------------------------------------------------------------------------------------
gpt-4o                      airline    200    50      4    0.420   0.42        11   30      0.2104  OK
gpt-4o                      retail     460   115      4    0.604   0.60        13   30      0.2858  OK
claude-3-5-sonnet-20241022  airline    400    50      8    0.460   0.46        12   30      0.2148  OK
claude-3-5-sonnet-20241022  retail     920   115      8    0.692   0.69        13   30      0.2544  OK
```

`paper` is the published success rate; `cost p50 $` is the median agent cost per run at
the main price.
