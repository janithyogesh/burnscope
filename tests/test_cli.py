"""Tests for the burnscope command-line interface."""

import pytest
from typer.testing import CliRunner

from burnscope import __version__
from burnscope.cli import app

runner = CliRunner()


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version(flag: str) -> None:
    result = runner.invoke(app, [flag])

    assert result.exit_code == 0
    assert result.output.strip() == f"burnscope {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage" in result.output
    assert "--version" in result.output
