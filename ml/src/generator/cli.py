"""CLI for the synthetic UPI event generator.

    python -m ml.src.generator.cli --days 90 --payers 5000 --seed 42 --out data/synthetic/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.generator.generator import generate, write_outputs  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic UPI events.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--payers", type=int, default=5000)
    parser.add_argument("--payees", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud-rate", type=float, default=0.007,
                         help="Target overall fraud rate (default 0.007 = 0.7%%, realistic band is 0.4%%-1.2%%).")
    parser.add_argument("--out", type=str, default="data/synthetic/")
    args = parser.parse_args(argv)

    events, summary = generate(
        days=args.days,
        n_payers=args.payers,
        n_payees=args.payees,
        seed=args.seed,
        fraud_rate=args.fraud_rate,
    )

    out_dir = Path(args.out)
    jsonl_path, csv_path = write_outputs(events, out_dir)

    print(f"Wrote {summary['event_count']} events to:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
