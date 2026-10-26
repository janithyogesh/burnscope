"""Command-line interface for burnscope."""

from typing import Annotated

import typer

from burnscope import __version__

app = typer.Typer(
    name="burnscope",
    help="Find where your AI agent burns money, and how to fix it.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print the installed version and exit when --version is passed."""
    if value:
        typer.echo(f"burnscope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the burnscope version and exit.",
        ),
    ] = False,
) -> None:
    """Find where your AI agent burns money, and how to fix it."""
