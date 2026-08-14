"""Command line entry point: python -m cts <command>.

Every handler imports its stage module lazily, so a module that is missing or
fails to import only breaks its own command.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config, load_config


def _ingest(cfg: Config, args: argparse.Namespace) -> None:
    # imported in pipeline order, so a missing module names the stage that stops
    from .ingest import run as run_ingest
    from .edhrec import run as run_edhrec
    from .power import run as run_power
    from .art import run as run_art

    run_ingest(cfg)
    run_edhrec(cfg)
    run_power(cfg)
    run_art(cfg)


def _describe(cfg: Config, args: argparse.Namespace) -> None:
    from .describe import run as run_describe

    run_describe(cfg, limit=args.limit, backfill_stale=args.backfill_stale)


def _embed(cfg: Config, args: argparse.Namespace) -> None:
    from .embed import run as run_embed

    run_embed(cfg)


def _search(cfg: Config, args: argparse.Namespace) -> None:
    from .search import run as run_search

    run_search(
        cfg,
        args.query,
        band=args.band,
        colors=args.colors,
        k=args.k,
        as_json=args.as_json,
    )


def _refresh(cfg: Config, args: argparse.Namespace) -> None:
    from .refresh import run as run_refresh

    sys.exit(run_refresh(cfg))


def _eval(cfg: Config, args: argparse.Namespace) -> None:
    from .evaluate import run as run_evaluate

    run_evaluate(cfg, collect_prefs=args.collect_prefs)


def _synth(cfg: Config, args: argparse.Namespace) -> None:
    from .synth import run as run_synth

    run_synth(cfg, limit=args.limit)


def _export_training(cfg: Config, args: argparse.Namespace) -> None:
    from .export_training import run as run_export

    run_export(cfg, target=args.target, out=args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cts",
        description="Scrying Pool — search Magic commanders by what their art depicts, "
        "means, or evokes.",
        epilog='example: python -m cts search "commanders that look lonely" --band 3',
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        metavar="PATH",
        help="config file to load (default: config.toml)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser(
        "ingest",
        help="phases 1-4: Scryfall bulk, EDHREC, power scores, art downloads",
        description="Run the data pipeline end to end. Resumable: every stage skips "
        "rows it has already done.",
    )
    p.set_defaults(handler=_ingest)

    p = sub.add_parser(
        "describe",
        help="phase 5: vision pass over art crops",
        description="Describe each downloaded artwork in two layers, default printings "
        "first. Hours of work; interrupt and re-run freely.",
    )
    p.add_argument("--limit", type=int, default=None, metavar="N", help="stop after N artworks")
    p.add_argument(
        "--backfill-stale",
        action="store_true",
        help="also re-describe artworks written by an older prompt_version",
    )
    p.set_defaults(handler=_describe)

    p = sub.add_parser(
        "embed",
        help="phase 6: embed every proposition",
        description="Embed propositions that have no vector yet and store them as "
        "float32 blobs.",
    )
    p.set_defaults(handler=_embed)

    p = sub.add_parser(
        "search",
        help="phases 7-8: answer a theme query",
        description="Route, expand, retrieve, judge, and verify with eyes; print a pool "
        "of commanders with the printing that earned the match.",
    )
    p.add_argument("query", metavar="QUERY", help="free-text theme, in quotes")
    p.add_argument(
        "--band",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="power band, 1 weakest to 5 strongest (default: any)",
    )
    p.add_argument(
        "--colors",
        default=None,
        metavar="WUBRG",
        help='color identity filter, e.g. "WUB": results must fit inside these colors',
    )
    p.add_argument("-k", "--k", type=int, default=5, metavar="N", help="results to return (default: 5)")
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit the full pool as JSON, including all links",
    )
    p.set_defaults(handler=_search)

    p = sub.add_parser(
        "refresh",
        help="phase 10: weekly idempotent update",
        description="Preflight Ollama, then pick up new cards, new artwork, and moved "
        "EDHREC numbers. Exits non-zero on failure.",
    )
    p.set_defaults(handler=_refresh)

    p = sub.add_parser(
        "eval",
        help="phase 11: run the held-out query set",
        description="Score eval/queries.jsonl: recall on literal themes, precision at 5 "
        "on abstract ones, pairwise agreement, latency.",
    )
    p.add_argument(
        "--collect-prefs",
        action="store_true",
        help="prompt for pairwise preferences on abstract themes and store them",
    )
    p.set_defaults(handler=_eval)

    p = sub.add_parser(
        "synth",
        help="phase 12: generate the synthetic theme corpus",
        description="Ask the judge model for themes each artwork genuinely satisfies, "
        "plus near misses, as training data.",
    )
    p.add_argument("--limit", type=int, default=None, metavar="N", help="stop after N artworks")
    p.set_defaults(handler=_synth)

    p = sub.add_parser(
        "export-training",
        help="phase 12: write a JSONL training set",
        description="Export the accumulated queries, retrievals, judgments and "
        "preferences as training data.",
    )
    p.add_argument(
        "--target",
        required=True,
        choices=["embed", "judge"],
        help="which dataset to write: embed (contrastive triples) or judge (fit labels)",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="output directory, receiving <target>_train.jsonl and <target>_val.jsonl "
        "(default: exports/)",
    )
    p.set_defaults(handler=_export_training)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        raise SystemExit(2)
    cfg = load_config(args.config)
    args.handler(cfg, args)


if __name__ == "__main__":
    main()
