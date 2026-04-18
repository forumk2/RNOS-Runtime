from __future__ import annotations

from pathlib import Path

import click

from .config import load_config
from .indexer import run_index
from .searcher import search
from .synthesizer import synthesize, _EXPLORE_SYSTEM_PROMPT, _EXPLORE_OUTPUT_RESERVE

_EXPLORE_TOP_K = 4


@click.group()
def cli() -> None:
    """Semantic search and Q&A over the RNOS-Runtime codebase."""


@cli.command()
@click.option(
    "--root",
    default=".",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Repo root (default: current directory).",
)
def index(root: Path) -> None:
    """Walk repo, chunk files, embed, and upsert into .rnos-query/index.db."""
    cfg = load_config()
    run_index(root.resolve(), cfg)


@cli.command()
@click.argument("query")
def ask(query: str) -> None:
    """Answer a question using semantically retrieved chunks from the index."""
    cfg = load_config()
    results = search(query, cfg)
    if not results:
        click.echo("No relevant chunks found. Run `rnos-query index` first.")
        return
    answer = synthesize(query, results, cfg)
    click.echo(answer)
    click.echo("\nCitations:")
    for r in results:
        sha = r.commit_sha or "unknown"
        click.echo(f"  {r.path}:{r.start_line}-{r.end_line}  [{sha}]")


@cli.command()
@click.argument("query")
def explore(query: str) -> None:
    """Explore a query with structured grounded/inferred/proposed output."""
    cfg = load_config()
    results = search(query, cfg, top_k=_EXPLORE_TOP_K)
    if not results:
        click.echo("No relevant chunks found. Run `rnos-query index` first.")
        return
    answer = synthesize(
        query,
        results,
        cfg,
        system_prompt=_EXPLORE_SYSTEM_PROMPT,
        output_reserve=_EXPLORE_OUTPUT_RESERVE,
    )
    click.echo(answer)
    click.echo("\nCitations:")
    for r in results:
        sha = r.commit_sha or "unknown"
        click.echo(f"  {r.path}:{r.start_line}-{r.end_line}  [{sha}]")


@cli.command()
@click.argument("query")
def debug(query: str) -> None:
    """Retrieve chunks with scores; skip synthesis."""
    cfg = load_config()
    results = search(query, cfg)
    if not results:
        click.echo("No results. Run `rnos-query index` first.")
        return
    for r in results:
        sha = r.commit_sha or "unknown"
        click.echo(
            f"\n--- {r.path}:{r.start_line}-{r.end_line}  [{sha}]"
            f"  type={r.chunk_type}  score={r.score:.4f} ---"
        )
        preview = r.content[:600]
        if len(r.content) > 600:
            preview += "\n[... truncated]"
        click.echo(preview)
