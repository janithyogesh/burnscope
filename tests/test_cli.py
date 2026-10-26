"""Tests for the burnscope command-line interface."""

import re

import pytest
from typer.testing import CliRunner

from burnscope import __version__
from burnscope.cli import app

runner = CliRunner()

# Rich adds ANSI colour codes when it detects CI (e.g. GITHUB_ACTIONS=true).
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI colour codes so assertions don't depend on the terminal."""
    return ANSI_ESCAPE.sub("", text)


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version(flag: str) -> None:
    result = runner.invoke(app, [flag])

    assert result.exit_code == 0
    assert plain(result.output).strip() == f"burnscope {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    output = plain(result.output)

    assert result.exit_code == 2
    assert "Usage" in output
    assert "--version" in output
