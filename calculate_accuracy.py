#!/usr/bin/env python3
"""Calculate accuracy metrics for results."""

import json
import argparse
from collections import defaultdict


def load_results(filepath: str) -> list[dict]:
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_accuracy(results: list[dict]) -> dict:
    """Calculate overall and per-category accuracy metrics."""
    metrics = {
        "overall": {"correct": 0, "total": 0, "errors": 0},
        "by_structure": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_view": defaultdict(lambda: {"correct": 0, "total": 0}),
    }

    for item in results:
        # Skip items with errors
        if item.get("error", False):
            metrics["overall"]["errors"] += 1
            continue

        metrics["overall"]["total"] += 1
        structure = item.get("structure", "Unknown")
        view = item.get("view", "Unknown")

        metrics["by_structure"][structure]["total"] += 1
        metrics["by_view"][view]["total"] += 1

        if item.get("is_correct", False):
            metrics["overall"]["correct"] += 1
            metrics["by_structure"][structure]["correct"] += 1
            metrics["by_view"][view]["correct"] += 1

    return metrics


def print_metrics(metrics: dict) -> None:
    """Print accuracy metrics in a formatted way."""
    overall = metrics["overall"]
    total = overall["total"]
    correct = overall["correct"]
    errors = overall["errors"]

    print("=" * 60)
    print("MEDGEMMA RESULTS ACCURACY REPORT")
    print("=" * 60)

    # Overall accuracy
    print("\n📊 OVERALL ACCURACY")
    print("-" * 40)
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Correct:    {correct:>6} / {total:<6} ({accuracy:.2f}%)")
    else:
        print("No valid samples to evaluate.")

    if errors > 0:
        print(f"Errors:     {errors:>6} (excluded from accuracy)")

    # Accuracy by structure
    print("\n📋 ACCURACY BY STRUCTURE")
    print("-" * 40)
    by_structure = metrics["by_structure"]
    structure_data = []
    for structure, data in sorted(by_structure.items()):
        if data["total"] > 0:
            acc = (data["correct"] / data["total"]) * 100
            structure_data.append((structure, data["correct"], data["total"], acc))

    # Sort by accuracy descending
    structure_data.sort(key=lambda x: x[3], reverse=True)
    for structure, correct, total, acc in structure_data:
        print(f"{structure:<35} {correct:>4}/{total:<4} ({acc:>6.2f}%)")

    # Accuracy by view
    print("\n🔍 ACCURACY BY VIEW")
    print("-" * 40)
    by_view = metrics["by_view"]
    view_data = []
    for view, data in sorted(by_view.items()):
        if data["total"] > 0:
            acc = (data["correct"] / data["total"]) * 100
            view_data.append((view, data["correct"], data["total"], acc))

    # Sort by accuracy descending
    view_data.sort(key=lambda x: x[3], reverse=True)
    for view, correct, total, acc in view_data:
        print(f"{view:<35} {correct:>4}/{total:<4} ({acc:>6.2f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy metrics for MedGemma results"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="outputs/medgemma_results.json",
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON instead of formatted text",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    metrics = calculate_accuracy(results)

    if args.json:
        # Convert defaultdicts to regular dicts for JSON serialization
        output = {
            "overall": metrics["overall"],
            "by_structure": dict(metrics["by_structure"]),
            "by_view": dict(metrics["by_view"]),
        }
        # Add accuracy percentages
        if output["overall"]["total"] > 0:
            output["overall"]["accuracy"] = (
                output["overall"]["correct"] / output["overall"]["total"]
            ) * 100
        for category in ["by_structure", "by_view"]:
            for key, data in output[category].items():
                if data["total"] > 0:
                    data["accuracy"] = (data["correct"] / data["total"]) * 100
        print(json.dumps(output, indent=2))
    else:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
