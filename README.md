# burnscope

[![CI](https://github.com/janithyogesh/burnscope/actions/workflows/ci.yml/badge.svg)](https://github.com/janithyogesh/burnscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Find where your AI agent burns money, and how to fix it.**

> **Status: early development.** Nothing below is usable yet except `burnscope --version`.
> This README describes what Version 1 will do.

## What it will do

Point burnscope at an agent and a small set of sample tasks. It runs each task several
times on a few models, records every model call, checks which runs succeeded, and
produces a report with:

- **Waste findings from real recorded runs.** Each finding has a cause, a suggested fix
  and an estimated saving. For example:
  - prompt cache missed on most calls because something that changes (like a timestamp)
    sits at the start of the prompt
  - the same tool called with the same arguments several times in one run
  - runs hitting the output token limit with no final answer
  - steps where a cheaper model would likely be enough
- **Cost per successful task** per model, including retries and failed runs.
- **Cost spread:** median (p50) and worst-case (p90) cost per run.
- **Token breakdown:** uncached input, cached input, cache writes, output and reasoning
  tokens.

Version 1 works with free API tiers (Groq, Google AI Studio, local Ollama) and shows what
the same runs *would* cost at published paid rates.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/janithyogesh/burnscope.git
cd burnscope
uv sync                     # create .venv and install everything

uv run burnscope --version  # try the CLI
uv run pytest               # tests
uv run ruff check           # lint
uv run ruff format          # format
uv run mypy                 # type check (strict)
```

## License

[MIT](LICENSE)
