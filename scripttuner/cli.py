"""ScriptTuner CLI 진입점.

사용 예:
    uv run scripttuner download sbcsae
    uv run scripttuner download switchboard --force
    uv run scripttuner parse sbcsae SBC016
    uv run scripttuner clean sbcsae SBC016
    uv run scripttuner monologue sbcsae SBC016
    uv run scripttuner sample ami --target 2000 --seed 42
    uv run scripttuner pairs sbcsae SBC016 --model deepseek/deepseek-v4-flash
    uv run scripttuner run switchboard --all --through monologue
    uv run scripttuner --help

`pairs` 서브커맨드는 `.env`에서 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 자동 인식.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripttuner.corpora import REGISTRY
from scripttuner.llm.rate_limit import RateLimitRetryClient
from scripttuner.persistence.jsonl import read_jsonl, write_jsonl
from scripttuner.preprocessing.ir import Monologue, Pair, Utterance
from scripttuner.preprocessing.monologue import DEFAULT_MIN_TOKENS, build_monologues
from scripttuner.preprocessing.pairs import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_STYLE,
    convert_to_formal,
)
from scripttuner.preprocessing.stats import compute_stats
from scripttuner.training.formatters import format_split_folder
from scripttuner.training.registry import MODEL_KEYS
from scripttuner.training.sample import sample_balanced
from scripttuner.training.sample import stem_of as sample_stem_of
from scripttuner.training.split import split_by_speaker, write_split_files

DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_DATA_DIR = Path("data")
RUN_STAGES = ("parse", "clean", "monologue", "pairs", "stats")
"""Ordered pipeline stages the `run` orchestrator can execute."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripttuner", description="ScriptTuner CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download", help="Download a dataset by name.")
    dl.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus to download.")
    dl.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Base destination directory (default: {DEFAULT_DATASETS_DIR}).",
    )
    dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )

    pr = subparsers.add_parser(
        "parse", help="Parse a corpus stem into parsed Utterance JSONL."
    )
    pr.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (selects adapter).")
    pr.add_argument("stem", help="File stem (e.g. SBC016 or sw2005).")
    pr.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Source corpus directory base (default: {DEFAULT_DATASETS_DIR}).",
    )
    pr.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )

    cl = subparsers.add_parser(
        "clean", help="Apply marker cleaning to parsed Utterance JSONL."
    )
    cl.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (selects cleaner).")
    cl.add_argument("stem", help="File stem (e.g. SBC016).")
    cl.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )

    mn = subparsers.add_parser(
        "monologue", help="Build Monologues from cleaned Utterance JSONL."
    )
    mn.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (resolves source dir).")
    mn.add_argument("stem", help="File stem (e.g. SBC016).")
    mn.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    mn.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"Minimum word tokens per monologue (default: {DEFAULT_MIN_TOKENS}).",
    )

    pa = subparsers.add_parser(
        "pairs",
        help="Convert Monologues to (formal, spoken) Pair JSONL via LLM.",
    )
    pa.add_argument(
        "corpus", choices=sorted(REGISTRY), help="Corpus name (resolves source dir)."
    )
    pa.add_argument(
        "stem",
        nargs="?",
        help="File stem (e.g. SBC016). Omit when using --all.",
    )
    pa.add_argument(
        "--all",
        dest="all_stems",
        action="store_true",
        help=(
            "Process every monologue stem under the input subdir "
            "(<data-dir>/<monologues-subdir>/<SOURCE>/), reusing one LLM client."
        ),
    )
    pa.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL"),
        help="LLM model slug (required unless LLM_MODEL env is set).",
    )
    pa.add_argument(
        "--model-alias",
        default=os.environ.get("LLM_MODEL_ALIAS"),
        help=(
            "Cache-key identity. Use a stable name shared across routing variants "
            "of the same weights (e.g. omit `:free`/`:nitro` suffixes) so the "
            "cache hits across routes. Defaults to LLM_MODEL_ALIAS env, then to "
            "--model when unset (legacy behavior)."
        ),
    )
    pa.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        help=f"Style label for produced pairs (default: {DEFAULT_STYLE}).",
    )
    pa.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help=f"Prompt version identifier (default: {DEFAULT_PROMPT_VERSION}).",
    )
    pa.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    pa.add_argument(
        "--monologues-subdir",
        default="monologues",
        help=(
            "Input monologue subdir under <data-dir> (default: monologues). "
            "Use 'monologues_sampled' to run pairs over a `sample` subset."
        ),
    )
    pa.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory (default: <data-dir>/cache/pairs).",
    )
    pa.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk cache.",
    )
    pa.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide tqdm progress bar.",
    )
    pa.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="OpenAI SDK transient retry count (default: 3).",
    )
    pa.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N monologues (default: all).",
    )
    pa.add_argument(
        "--repunct",
        action="store_true",
        help=(
            "Apply Tier-B re-punctuation (commas/question marks) to the spoken "
            "side using the same model (cf. ADR-0012). Formal side and its cache "
            "are unaffected. Invariant failures fall back to Tier-A text."
        ),
    )

    sm = subparsers.add_parser(
        "sample",
        help=(
            "Speaker-aware, meeting-type-balanced subsample of monologues "
            "(scale-match to casual; cf. ADR-0013)."
        ),
    )
    sm.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    sm.add_argument(
        "--target",
        type=int,
        default=2000,
        help="Target monologue count (split 1:1 across meeting types; default: 2000).",
    )
    sm.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Speaker-shuffle seed for reproducibility (default: 42).",
    )
    sm.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    sm.add_argument(
        "--output-subdir",
        default="monologues_sampled",
        help="Output subdir under <data-dir> (default: monologues_sampled).",
    )

    st = subparsers.add_parser(
        "stats", help="Compute aggregate statistics over Pair JSONL."
    )
    st.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (resolves source dir).")
    st.add_argument("stem", help="File stem (e.g. SBC016).")
    st.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    st.add_argument(
        "--no-pos",
        action="store_true",
        help="Skip POS-based stats (lexical density, phrasal verbs).",
    )

    ag = subparsers.add_parser(
        "aggregate",
        help="Concat per-stem Pair JSONLs into _all.jsonl + corpus-wide stats.",
    )
    ag.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    ag.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    ag.add_argument(
        "--no-pos",
        action="store_true",
        help="Skip POS-based stats in the aggregate.",
    )

    sp = subparsers.add_parser(
        "split",
        help="Create speaker-aware train/validation/test splits from Pair JSONL.",
    )
    sp.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    sp.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input Pair JSONL (default: <data-dir>/pairs/<SOURCE>/_all.jsonl).",
    )
    sp.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    sp.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output split directory "
            "(default: <data-dir>/finetune/<SOURCE>/splits)."
        ),
    )
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--train-ratio", type=float, default=0.8)
    sp.add_argument("--validation-ratio", type=float, default=0.1)
    sp.add_argument("--test-ratio", type=float, default=0.1)

    skt = subparsers.add_parser(
        "spokenness-train",
        help="Train the P(spoken) classifier on the train split (cf. ADR-0014).",
    )
    skt.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    skt.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    skt.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Input split directory (default: <data-dir>/finetune/<SOURCE>/splits).",
    )
    skt.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output model path "
        "(default: <data-dir>/finetune/<SOURCE>/spokenness/clf.joblib).",
    )
    skt.add_argument("--seed", type=int, default=42)

    fm = subparsers.add_parser(
        "format",
        help="Format fine-tuning splits for a target model family.",
    )
    fm.add_argument("model_key", choices=sorted(MODEL_KEYS), help="Target model key.")
    fm.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    fm.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    fm.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Input split directory (default: <data-dir>/finetune/<SOURCE>/splits).",
    )
    fm.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output formatted directory "
            "(default: <data-dir>/finetune/<SOURCE>/formatted/<model_key>)."
        ),
    )

    tr = subparsers.add_parser(
        "train",
        help="QLoRA fine-tune a chat model (Unsloth) on formatted splits.",
    )
    tr.add_argument("model_key", choices=sorted(MODEL_KEYS), help="Target model key.")
    tr.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    tr.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    tr.add_argument(
        "--formatted-dir",
        type=Path,
        default=None,
        help="Formatted split dir (default: <data-dir>/finetune/<SOURCE>/formatted/<model_key>).",
    )
    tr.add_argument(
        "--run-name", default=None, help="Run name (default: <model_key>-<SOURCE>-lora)."
    )
    tr.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: runs/finetune/<run-name>).",
    )
    tr.add_argument("--max-seq-length", type=int, default=2048)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=16)
    tr.add_argument("--lora-dropout", type=float, default=0.0)
    tr.add_argument("--epochs", type=float, default=1.0)
    tr.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N steps instead of epochs (smoke/quick runs).",
    )
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--learning-rate", type=float, default=2e-4)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute activations to save VRAM (default: on). "
        "Use --no-gradient-checkpointing to trade VRAM for ~20-30%% speed when memory allows.",
    )

    gn = subparsers.add_parser(
        "generate",
        help="Generate predictions from a trained adapter on a formatted split.",
    )
    gn.add_argument("model_key", choices=sorted(MODEL_KEYS), help="Target model key.")
    gn.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    gn.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    gn.add_argument(
        "--run-name", default=None, help="Run name (default: <model_key>-<SOURCE>-lora)."
    )
    gn.add_argument(
        "--adapter-dir",
        default=None,
        help="Local adapter dir or an HF repo id "
        "(default: runs/finetune/<run-name>/adapter).",
    )
    gn.add_argument(
        "--formatted-dir",
        type=Path,
        default=None,
        help="Formatted split dir (default: <data-dir>/finetune/<SOURCE>/formatted/<model_key>).",
    )
    gn.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Predictions JSONL (default: runs/eval/<run-name>/predictions.jsonl).",
    )
    gn.add_argument(
        "--split", default="test", choices=["train", "validation", "test"], help="Split to run."
    )
    gn.add_argument("--limit", type=int, default=None, help="Only generate for the first N rows.")
    gn.add_argument("--max-new-tokens", type=int, default=256)
    gn.add_argument("--batch-size", type=int, default=8)
    gn.add_argument("--max-seq-length", type=int, default=2048)
    gn.add_argument("--repetition-penalty", type=float, default=1.0)
    gn.add_argument("--no-repeat-ngram-size", type=int, default=0)

    ev = subparsers.add_parser(
        "evaluate",
        help="Compute spoken-ness metrics (length/filler/pause/lexical density) for predictions.",
    )
    ev.add_argument(
        "--predictions", type=Path, required=True, help="predictions.jsonl from `generate`."
    )
    ev.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics JSON (default: metrics.json beside the predictions file).",
    )
    ev.add_argument(
        "--no-pos", action="store_true", help="Skip POS-based metrics (lexical density)."
    )
    ev.add_argument(
        "--spokenness-model",
        type=Path,
        default=None,
        help="Trained spokenness classifier (.joblib from `spokenness-train`). "
        "If given, adds the input->prediction P(spoken) delta block (cf. ADR-0014).",
    )

    pl = subparsers.add_parser(
        "plot",
        help="Plot training curves (train/eval loss) from log_history.json.",
    )
    pl.add_argument("model_key", choices=sorted(MODEL_KEYS), help="Target model key.")
    pl.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    pl.add_argument(
        "--run-name",
        default=None,
        help="Run name (default: <model_key>-<SOURCE>-lora).",
    )
    pl.add_argument(
        "--log-history",
        type=Path,
        default=None,
        help="log_history.json (default: runs/finetune/<run-name>/log_history.json).",
    )
    pl.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: runs/finetune/<run-name>/training_curves.png).",
    )

    rn = subparsers.add_parser(
        "run",
        help="End-to-end pipeline: parse > clean > monologue > pairs > stats. "
        "Accepts one or more stems, or --all for every stem in the corpus dir. "
        "Use --through to stop early (e.g. before the LLM pairs stage).",
    )
    rn.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    rn.add_argument(
        "stems",
        nargs="*",
        help="One or more file stems (e.g. SBC016 SBC017). Omit when using --all.",
    )
    rn.add_argument(
        "--all",
        dest="all_stems",
        action="store_true",
        help="Process every stem under <datasets-dir>/<corpus>/ (per the adapter).",
    )
    rn.add_argument(
        "--through",
        choices=RUN_STAGES,
        default="stats",
        help="Run stages up to and including this one (default: stats). "
        "Use 'monologue' to stop before the LLM pairs stage.",
    )
    rn.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Source corpus directory base (default: {DEFAULT_DATASETS_DIR}).",
    )
    rn.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    rn.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL"),
        help="LLM model slug (required unless LLM_MODEL env is set).",
    )
    rn.add_argument(
        "--model-alias",
        default=os.environ.get("LLM_MODEL_ALIAS"),
        help="Cache-key identity (env LLM_MODEL_ALIAS). See `pairs` subcommand for details.",
    )
    rn.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    rn.add_argument("--limit", type=int, default=None, help="Limit pairs to first N monologues.")
    rn.add_argument("--no-progress", action="store_true")
    rn.add_argument("--no-cache", action="store_true")
    rn.add_argument("--max-retries", type=int, default=3)
    rn.add_argument("--no-pos", action="store_true", help="Skip POS-based stats.")

    return parser


def _source_name(corpus: str) -> str:
    return REGISTRY[corpus].source_name


def _run_download(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    dest_dir = args.dest / args.corpus
    files = adapter.download(dest_dir, force=args.force)
    print(f"OK: {len(files)} files in {dest_dir}")
    return 0


def _run_parse(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    corpus_dir = args.datasets_dir / args.corpus
    utterances = adapter.parse_stem(corpus_dir, args.stem)
    out_path = args.data_dir / "parsed" / adapter.source_name / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, utterances)
    print(f"OK: wrote {n} utterances to {out_path}")
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    in_path = args.data_dir / "parsed" / adapter.source_name / f"{args.stem}.jsonl"
    utterances = read_jsonl(in_path, Utterance)
    cleaned = adapter.clean(utterances)
    out_path = args.data_dir / "cleaned" / adapter.source_name / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, cleaned)
    print(f"OK: wrote {n} cleaned utterances to {out_path}")
    return 0


def _run_monologue(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    in_path = args.data_dir / "cleaned" / adapter.source_name / f"{args.stem}.jsonl"
    utterances = read_jsonl(in_path, Utterance)
    monologues = build_monologues(
        utterances,
        min_tokens=args.min_tokens,
        backchannel_words=adapter.backchannel_words,
    )
    out_path = args.data_dir / "monologues" / adapter.source_name / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, monologues)
    print(f"OK: wrote {n} monologues to {out_path}")
    return 0


def _provider_routing_from_env() -> dict[str, Any] | None:
    """Build OpenRouter provider-routing prefs from env vars, or None.

    - ``LLM_PROVIDER_SORT``        → provider.sort (e.g. "price")
    - ``LLM_PROVIDER_MAX_PROMPT``  → provider.max_price.prompt  ($/M input tokens)
    - ``LLM_PROVIDER_MAX_COMPLETION`` → provider.max_price.completion ($/M output)

    Ignored by non-OpenRouter endpoints. Returns None when none are set so the
    request body stays clean.
    """
    provider: dict[str, Any] = {}
    sort = os.environ.get("LLM_PROVIDER_SORT")
    if sort:
        provider["sort"] = sort
    max_price: dict[str, float] = {}
    if (p := os.environ.get("LLM_PROVIDER_MAX_PROMPT")) is not None:
        max_price["prompt"] = float(p)
    if (c := os.environ.get("LLM_PROVIDER_MAX_COMPLETION")) is not None:
        max_price["completion"] = float(c)
    if max_price:
        provider["max_price"] = max_price
    return provider or None


def _build_llm_client(model: str, max_retries: int) -> RateLimitRetryClient:
    """Build the LLM client.

    All free-tier OpenAI-compatible endpoints (OpenRouter, Groq, …) hit
    per-minute caps that the SDK's 8s-capped backoff cannot span, so every
    provider gets the RPM-cap-aware `RateLimitRetryClient` (sleeps to the
    Retry-After / 60s window, then one retry). ``max_retries`` is unused now;
    the wrapper owns retry policy and disables the SDK's internal loop.

    OpenRouter provider routing (cheapest-provider selection) is read from env
    via `_provider_routing_from_env` and passed through to the request body.
    """
    return RateLimitRetryClient(model=model, provider=_provider_routing_from_env())


def _run_pairs(args: argparse.Namespace) -> int:
    if not args.model:
        print(
            "error: --model is required (or set LLM_MODEL environment variable)",
            file=sys.stderr,
        )
        return 2
    source = _source_name(args.corpus)
    subdir = getattr(args, "monologues_subdir", "monologues")
    mono_dir = args.data_dir / subdir / source

    all_stems = getattr(args, "all_stems", False)
    if all_stems:
        if getattr(args, "stem", None):
            print("error: provide a stem OR --all, not both.", file=sys.stderr)
            return 2
        stems = sorted(
            p.stem for p in mono_dir.glob("*.jsonl") if not p.name.startswith("_")
        )
        if not stems:
            print(f"error: no monologue jsonls under {mono_dir}", file=sys.stderr)
            return 1
    else:
        if not getattr(args, "stem", None):
            print("error: provide a stem or use --all.", file=sys.stderr)
            return 2
        stems = [args.stem]

    cache_dir: Path | None
    if args.no_cache:
        cache_dir = None
    else:
        cache_dir = args.cache_dir or (args.data_dir / "cache" / "pairs")

    # Build the client(s) once and reuse across all stems.
    client = _build_llm_client(args.model, args.max_retries)
    repunct_client = None
    repunct_cache_dir = None
    if getattr(args, "repunct", False):
        repunct_client = _build_llm_client(args.model, args.max_retries)
        # repunct 캐시는 formal과 별도 디렉토리 (--cache-dir은 formal 전용).
        repunct_cache_dir = None if args.no_cache else (args.data_dir / "cache" / "repunct")

    total = 0
    for stem in stems:
        monologues = read_jsonl(mono_dir / f"{stem}.jsonl", Monologue)
        if args.limit is not None:
            monologues = monologues[: args.limit]
        pairs = convert_to_formal(
            monologues,
            client=client,
            model=args.model,
            model_alias=args.model_alias,
            cache_dir=cache_dir,
            prompt_version=args.prompt_version,
            style=args.style,
            progress=not args.no_progress,
            repunct_client=repunct_client,
            repunct_cache_dir=repunct_cache_dir,
        )
        out_path = args.data_dir / "pairs" / source / f"{stem}.jsonl"
        n = write_jsonl(out_path, pairs)
        total += n
        skipped = len(monologues) - n
        print(f"OK: wrote {n} pairs to {out_path} ({skipped} skipped)")
    if all_stems:
        print(f"[pairs] summary: {len(stems)} stems, {total} pairs", file=sys.stderr)
    return 0


def _meeting_type_map(parsed_dir: Path) -> dict[str, str]:
    """stem -> meeting_type, read from the first row of each parsed JSONL.

    Monologue records drop per-utterance metadata, so the scenario/nonscenario
    label is recovered from the parsed stage (cf. AMI parser metadata).
    """
    mapping: dict[str, str] = {}
    for path in parsed_dir.glob("*.jsonl"):
        for u in read_jsonl(path, Utterance):
            mapping[path.stem] = str(u.metadata.get("meeting_type", "unknown"))
            break
    return mapping


def _run_sample(args: argparse.Namespace) -> int:
    source = _source_name(args.corpus)
    mono_dir = args.data_dir / "monologues" / source
    if not mono_dir.is_dir():
        print(f"error: monologues dir not found: {mono_dir}", file=sys.stderr)
        return 1

    monologues: list[Monologue] = []
    for path in sorted(mono_dir.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        monologues.extend(read_jsonl(path, Monologue))
    if not monologues:
        print(f"error: no monologues under {mono_dir}", file=sys.stderr)
        return 1

    type_map = _meeting_type_map(args.data_dir / "parsed" / source)
    selected = sample_balanced(
        monologues, type_map, target=args.target, seed=args.seed
    )

    # Re-split selected monologues back into per-stem files (stem-centric contract).
    by_stem: dict[str, list[Monologue]] = defaultdict(list)
    for m in selected:
        by_stem[sample_stem_of(m.monologue_id)].append(m)
    out_dir = args.data_dir / args.output_subdir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, monos in sorted(by_stem.items()):
        write_jsonl(out_dir / f"{stem}.jsonl", monos)

    type_counts: dict[str, int] = defaultdict(int)
    for m in selected:
        type_counts[type_map.get(sample_stem_of(m.monologue_id), "unknown")] += 1
    print(
        f"OK: sampled {len(selected)} monologues "
        f"({dict(type_counts)}) across {len(by_stem)} stems to {out_dir}"
    )
    return 0


def _run_stats(args: argparse.Namespace) -> int:
    source = _source_name(args.corpus)
    in_path = args.data_dir / "pairs" / source / f"{args.stem}.jsonl"
    pairs = read_jsonl(in_path, Pair)
    result = compute_stats(pairs, include_pos=not args.no_pos)
    out_path = args.data_dir / "stats" / source / f"{args.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: wrote stats for {result['n_pairs']} pairs to {out_path}")
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    """Concat per-stem Pair JSONLs and compute corpus-wide stats.

    Writes two outputs:
      data/pairs/<SOURCE>/_all.jsonl     — all pairs concatenated
      data/stats/<SOURCE>/_aggregate.json — stats over _all.jsonl

    Files prefixed with ``_`` (incl. the previous _all.jsonl) are excluded from
    the input glob to keep aggregation idempotent.
    """
    source = _source_name(args.corpus)
    pairs_dir = args.data_dir / "pairs" / source
    if not pairs_dir.is_dir():
        print(f"error: pairs dir not found: {pairs_dir}", file=sys.stderr)
        return 1

    stem_paths = sorted(p for p in pairs_dir.glob("*.jsonl") if not p.name.startswith("_"))
    if not stem_paths:
        print(f"error: no per-stem pair jsonls under {pairs_dir}", file=sys.stderr)
        return 1

    pairs: list[Pair] = []
    for path in stem_paths:
        pairs.extend(read_jsonl(path, Pair))

    all_path = pairs_dir / "_all.jsonl"
    write_jsonl(all_path, pairs)
    print(f"OK: wrote {len(pairs)} pairs to {all_path}", file=sys.stderr)

    stats_dir = args.data_dir / "stats" / source
    stats_dir.mkdir(parents=True, exist_ok=True)
    result = compute_stats(pairs, include_pos=not args.no_pos)
    agg_path = stats_dir / "_aggregate.json"
    agg_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: wrote aggregate stats for {result['n_pairs']} pairs to {agg_path}")
    return 0


def _run_split(args: argparse.Namespace) -> int:
    source = _source_name(args.corpus)
    in_path = args.input or (args.data_dir / "pairs" / source / "_all.jsonl")
    out_dir = args.output_dir or (args.data_dir / "finetune" / source / "splits")
    pairs = read_jsonl(in_path, Pair)
    splits = split_by_speaker(
        pairs,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    manifest = write_split_files(
        splits,
        output_dir=out_dir,
        source_path=in_path,
        seed=args.seed,
        ratios=(args.train_ratio, args.validation_ratio, args.test_ratio),
    )
    print(f"OK: wrote fine-tuning splits to {out_dir} ({manifest['counts']})")
    return 0


def _run_spokenness_train(args: argparse.Namespace) -> int:
    from scripttuner.training.spokenness import save, train

    source = _source_name(args.corpus)
    splits_dir = args.splits_dir or (args.data_dir / "finetune" / source / "splits")
    out_path = args.output or (args.data_dir / "finetune" / source / "spokenness" / "clf.joblib")

    train_pairs = read_jsonl(splits_dir / "train.jsonl", Pair)
    model, metrics = train(train_pairs, seed=args.seed)
    save(model, out_path)
    (out_path.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"OK: trained spokenness classifier on {metrics['n_pairs']} pairs -> {out_path} "
        f"(holdout acc={metrics['holdout_accuracy']:.3f}, auc={metrics['holdout_auc']:.3f})"
    )
    return 0


def _run_format(args: argparse.Namespace) -> int:
    source = _source_name(args.corpus)
    splits_dir = args.splits_dir or (args.data_dir / "finetune" / source / "splits")
    out_dir = (
        args.output_dir
        or (args.data_dir / "finetune" / source / "formatted" / args.model_key)
    )
    manifest = format_split_folder(
        splits_dir=splits_dir,
        output_dir=out_dir,
        model_key=args.model_key,
    )
    print(
        f"OK: wrote {manifest['format']} fine-tuning data for "
        f"{args.model_key} to {out_dir} ({manifest['counts']})"
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    # Lazy import: Unsloth/torch are heavy and only needed for this command.
    from scripttuner.training.train import run_finetune

    source = _source_name(args.corpus)
    formatted_dir = args.formatted_dir or (
        args.data_dir / "finetune" / source / "formatted" / args.model_key
    )
    run_name = args.run_name or f"{args.model_key}-{source}-lora"
    output_dir = args.output_dir or (Path("runs") / "finetune" / run_name)
    manifest = run_finetune(
        model_key=args.model_key,
        formatted_dir=formatted_dir,
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(f"OK: trained {args.model_key} -> {output_dir} (adapter: {manifest['adapter_dir']})")
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    # Lazy import: torch/transformers are heavy and only needed for this command.
    from scripttuner.training.generate import run_generate

    source = _source_name(args.corpus)
    run_name = args.run_name or f"{args.model_key}-{source}-lora"
    adapter_dir = args.adapter_dir or (Path("runs") / "finetune" / run_name / "adapter")
    formatted_dir = args.formatted_dir or (
        args.data_dir / "finetune" / source / "formatted" / args.model_key
    )
    output_path = args.output or (Path("runs") / "eval" / run_name / "predictions.jsonl")
    summary = run_generate(
        model_key=args.model_key,
        adapter_dir=adapter_dir,
        formatted_dir=formatted_dir,
        output_path=output_path,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        limit=args.limit,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(f"OK: wrote {summary['n']} predictions to {output_path}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from scripttuner.training.evaluate import run_evaluate

    output_path = args.output or (args.predictions.parent / "metrics.json")
    metrics = run_evaluate(
        predictions_path=args.predictions,
        output_path=output_path,
        include_pos=not args.no_pos,
        spokenness_model_path=args.spokenness_model,
    )
    print(f"OK: wrote metrics for {metrics['n']} predictions to {output_path}")
    return 0


def _run_plot(args: argparse.Namespace) -> int:
    from scripttuner.training.plot import plot_training_curves

    source = _source_name(args.corpus)
    run_name = args.run_name or f"{args.model_key}-{source}-lora"
    log_history = args.log_history or (
        Path("runs") / "finetune" / run_name / "log_history.json"
    )
    output_path = args.output or (Path("runs") / "finetune" / run_name / "training_curves.png")
    summary = plot_training_curves(log_history_path=log_history, output_path=output_path)
    print(
        f"OK: wrote training curves to {output_path} "
        f"({summary['n_train_points']} train, {summary['n_eval_points']} eval points)"
    )
    return 0


def _resolve_run_stems(args: argparse.Namespace) -> tuple[list[str], int]:
    """Resolve the stem list for `run` from positional stems or --all.

    Returns (stems, rc). rc=0 on success; rc=2 on argument conflict; rc=1 when
    --all finds no stems in the corpus directory (per the adapter's enumerator).
    """
    if bool(args.stems) == bool(args.all_stems):
        print(
            "error: provide either one or more stems OR --all (not both, not neither).",
            file=sys.stderr,
        )
        return [], 2
    if args.all_stems:
        corpus_dir = args.datasets_dir / args.corpus
        stems = REGISTRY[args.corpus].enumerate_stems(corpus_dir)
        if not stems:
            print(f"error: no stems found under {corpus_dir}", file=sys.stderr)
            return [], 1
        return stems, 0
    return list(args.stems), 0


def _run_single_stem(stem: str, args: argparse.Namespace) -> int:
    """Process one stem through the pipeline up to and including args.through.

    Input presence is verified via the adapter's stem enumerator so the check is
    corpus-agnostic (CHAT = one .cha, Switchboard = a conversation's A/B files).
    """
    adapter = REGISTRY[args.corpus]
    source = adapter.source_name
    corpus_dir = args.datasets_dir / args.corpus
    if stem not in set(adapter.enumerate_stems(corpus_dir)):
        print(
            f"error: input not found for stem {stem!r} under {corpus_dir}\n"
            f"hint: run `scripttuner download {args.corpus}` first.",
            file=sys.stderr,
        )
        return 1

    base: dict[str, object] = {"corpus": args.corpus, "data_dir": args.data_dir}
    stage_kwargs: dict[str, dict[str, object]] = {
        "parse": {**base, "stem": stem, "datasets_dir": args.datasets_dir},
        "clean": {**base, "stem": stem},
        "monologue": {**base, "stem": stem, "min_tokens": args.min_tokens},
        "pairs": {
            **base,
            "stem": stem,
            "model": args.model,
            "model_alias": args.model_alias,
            "style": DEFAULT_STYLE,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "cache_dir": None,
            "no_cache": args.no_cache,
            "no_progress": args.no_progress,
            "max_retries": args.max_retries,
            "limit": args.limit,
        },
        "stats": {**base, "stem": stem, "no_pos": args.no_pos},
    }
    last = RUN_STAGES.index(args.through)
    for name in RUN_STAGES[: last + 1]:
        print(f"[run] {name} {args.corpus} {stem}", file=sys.stderr)
        rc = _COMMANDS[name](argparse.Namespace(**stage_kwargs[name]))
        if rc != 0:
            print(f"[run] {name} failed with rc={rc}", file=sys.stderr)
            return rc
    out_subdir = {"parse": "parsed", "clean": "cleaned", "monologue": "monologues",
                  "pairs": "pairs", "stats": "stats"}[args.through]
    ext = "json" if args.through == "stats" else "jsonl"
    print(
        f"[run] complete (through {args.through}): "
        f"data/{out_subdir}/{source}/{stem}.{ext}",
        file=sys.stderr,
    )
    return 0


def _run_run(args: argparse.Namespace) -> int:
    """End-to-end orchestrator. Multi-stem aware; per-stem failures are isolated."""
    stems, rc = _resolve_run_stems(args)
    if rc != 0:
        return rc

    succeeded: list[str] = []
    failed: list[str] = []
    for stem in stems:
        if _run_single_stem(stem, args) == 0:
            succeeded.append(stem)
        else:
            failed.append(stem)

    multi = len(stems) > 1 or args.all_stems
    if multi:
        summary = f"[run] summary: {len(succeeded)}/{len(stems)} succeeded"
        if failed:
            summary += f"; failed: {', '.join(failed)}"
        print(summary, file=sys.stderr)
    return 1 if failed else 0


_COMMANDS = {
    "download": _run_download,
    "parse": _run_parse,
    "clean": _run_clean,
    "monologue": _run_monologue,
    "sample": _run_sample,
    "pairs": _run_pairs,
    "stats": _run_stats,
    "aggregate": _run_aggregate,
    "split": _run_split,
    "spokenness-train": _run_spokenness_train,
    "format": _run_format,
    "train": _run_train,
    "generate": _run_generate,
    "evaluate": _run_evaluate,
    "plot": _run_plot,
    "run": _run_run,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # no-op when .env is missing
    parser = _build_parser()
    args = parser.parse_args(argv)
    runner = _COMMANDS.get(args.command)
    if runner is None:
        parser.error(f"Unknown command: {args.command}")
        return 2  # unreachable; parser.error exits
    return runner(args)


if __name__ == "__main__":
    sys.exit(main())
