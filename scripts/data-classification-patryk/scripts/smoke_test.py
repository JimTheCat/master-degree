"""Minimal API smoke test — classifies 1 speech, prints result, exits 0/1.

Use this BEFORE the full sweep to verify API keys, structured output parsing,
and token usage for paid models (Claude, Gemini).

Usage:
    python scripts/smoke_test.py --config configs/model_claude.yaml
    python scripts/smoke_test.py --config configs/model_gemini_flash.yaml
    python scripts/smoke_test.py --config configs/model_claude.yaml --speech-id ParlaMint-PL_2017-11-23-sejm-52-2.u526
    python scripts/smoke_test.py --config configs/model_claude.yaml --verbose
    python scripts/smoke_test.py --config configs/model_bielik.yaml --text "To skandal!"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.loader import load_gold_standard
from src.pipeline.runner import create_classifier, load_config

console = Console()

# USD per 1M tokens (input, output) — matches select_best_config pricing table.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    # claude-sonnet-5: $2/$10 intro pricing through 2026-08-31, then $3/$15.
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus": (5.00, 25.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (1.00, 5.00),
    # Gemini: thinking tokens bill as output.
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

# Env vars required per provider type.
PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "lmstudio": "",  # No key required — local server.
}


def _estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    for key, (in_p, out_p) in MODEL_PRICING.items():
        if key.lower() in model_name.lower():
            return (input_tokens / 1_000_000) * in_p + (output_tokens / 1_000_000) * out_p
    return 0.0


def _check_api_key(provider: str) -> None:
    """Exit with helpful message if required env var is missing."""
    env_var = PROVIDER_ENV_VARS.get(provider, "")
    if not env_var:
        return  # Local model — no key needed.
    value = os.environ.get(env_var, "")
    if not value:
        console.print(f"[bold red]FAIL Missing env var: {env_var}[/bold red]")
        console.print(f"  Add it to your .env file: {env_var}=sk-...")
        sys.exit(1)
    # Mask the key for display.
    masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
    console.print(f"[green]OK {env_var}[/green] = {masked}")
    if len(value) < 20:
        console.print(f"[yellow]WARN {env_var} is only {len(value)} chars — looks like a placeholder. Real keys are 100+ chars.[/yellow]")


async def run_smoke_test(
    config: dict, speech_id: str | None, text: str | None, verbose: bool
) -> bool:
    data_config = config["data"]
    model_config = config.get("model", {})
    provider = model_config.get("type", "openai")
    model_name = model_config.get("name", "?")

    console.print(f"\n[bold]Model:[/bold] {model_name}  [bold]Provider:[/bold] {provider}")
    _check_api_key(provider)

    if text:
        # Ad-hoc text — no gold labels, just verifies connection + parsing.
        from datetime import date

        from src.schema.models import AnnotatedSpeech, Speech

        entry = AnnotatedSpeech(speech=Speech(
            speech_id="smoke-test-adhoc",
            text=text,
            speaker="smoke-test",
            party="smoke-test",
            date=date.today(),
        ))
    else:
        # Load 1 speech from test.jsonl, falling back to the full gold standard
        # so any of the 500 speech ids works.
        test_path = Path(data_config["splits_dir"]) / "test.jsonl"
        test_data = load_gold_standard(test_path)
        if not test_data:
            console.print("[red]test.jsonl is empty[/red]")
            return False

        if speech_id:
            matches = [s for s in test_data if s.speech.speech_id == speech_id]
            if not matches:
                gold_data = load_gold_standard(Path(data_config["gold_standard_path"]))
                matches = [s for s in gold_data if s.speech.speech_id == speech_id]
            if not matches:
                console.print(
                    f"[red]speech_id {speech_id!r} not found in test.jsonl "
                    f"or gold standard[/red]"
                )
                return False
            entry = matches[0]
        else:
            entry = test_data[0]

    speech = entry.speech
    console.print(f"[bold]Speech:[/bold] {speech.speech_id}")
    console.print(f"[dim]Text preview:[/dim] {speech.text[:120]}…\n")

    # Build classifier (uses train data for few-shot if needed).
    train_data = None
    if config.get("prompt", {}).get("variant", "").startswith("fewshot"):
        train_path = Path(data_config["splits_dir"]) / "train.jsonl"
        train_data = load_gold_standard(train_path)

    classifier = create_classifier(config, train_data=train_data)
    console.print(f"[dim]Prompt variant:[/dim] {config.get('prompt', {}).get('variant', 'N/A')}")
    console.print("[bold yellow]Calling API...[/bold yellow]")

    prediction = await classifier.classify(speech)

    # Check for error in raw_response.
    if prediction.raw_response and prediction.raw_response.startswith(("ERROR:", "PARSE_FAILED")):
        console.print("\n[bold red]FAIL Classification failed[/bold red]")
        console.print(Panel(prediction.raw_response, title="Error", style="red"))
        return False
    if prediction.raw_response and prediction.raw_response.startswith("PARSE_SALVAGED"):
        console.print(
            "[yellow]WARN structured output failed validation — "
            "labels salvaged from raw text[/yellow]"
        )

    # Token usage & cost.
    input_tokens = prediction.token_usage.get("prompt_tokens", 0)
    output_tokens = prediction.token_usage.get("completion_tokens", 0)
    cost = _estimate_cost(model_name, input_tokens, output_tokens)

    # Results table.
    table = Table(title="Smoke test result", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", style="cyan")
    table.add_row("Latency", f"{prediction.latency_ms:.0f} ms" if prediction.latency_ms else "N/A")
    table.add_row("Input tokens", str(input_tokens))
    table.add_row("Output tokens", str(output_tokens))
    table.add_row("Estimated cost", f"${cost:.5f}")
    table.add_row("Labels detected", str(prediction.labels) if prediction.labels else "(none)")
    console.print(table)

    # Top-3 label scores.
    if prediction.label_scores:
        top3 = sorted(prediction.label_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        console.print("[dim]Top-3 scores:[/dim]  " +
                      "  ".join(f"{k}={v:.3f}" for k, v in top3))

    # Gold comparison (if available).
    if entry.gold:
        gold_labels = set(entry.gold.labels)
        pred_labels = set(prediction.labels)
        hit = gold_labels & pred_labels
        console.print(f"[dim]Gold labels:[/dim] {sorted(gold_labels)}")
        console.print(f"[dim]Overlap:[/dim] {sorted(hit)} ({len(hit)}/{len(gold_labels)})")

    if verbose and prediction.raw_response:
        console.print(Panel(prediction.raw_response[:1000], title="Raw response", style="dim"))

    # Validation checks.
    warnings = []
    if input_tokens == 0:
        warnings.append("input_tokens = 0 — token usage not captured (check LangChain version)")
    if cost > 0.10:
        warnings.append(f"cost ${cost:.4f} > $0.10 — unusually high for 1 speech")
    if prediction.latency_ms and prediction.latency_ms > 30_000:
        warnings.append(f"latency {prediction.latency_ms:.0f}ms > 30s — slow connection?")

    for w in warnings:
        console.print(f"[yellow]WARN {w}[/yellow]")

    console.print(f"\n[bold green]OK Smoke test PASSED — {model_name}[/bold green]")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test: 1 speech through an LLM API")
    parser.add_argument("--config", required=True, help="Model config, e.g. configs/model_claude.yaml")
    parser.add_argument(
        "--speech-id", default=None, help="Specific speech ID (test.jsonl or full gold standard)"
    )
    parser.add_argument(
        "--text", default=None, help="Classify this ad-hoc text instead of a dataset speech"
    )
    parser.add_argument("--verbose", action="store_true", help="Print raw API response")
    args = parser.parse_args()

    load_dotenv(override=True)
    config = load_config(args.config)

    try:
        success = asyncio.run(run_smoke_test(config, args.speech_id, args.text, args.verbose))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]FAIL Smoke test FAILED: {type(exc).__name__}: {exc}[/bold red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
