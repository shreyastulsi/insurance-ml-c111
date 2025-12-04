"""Utility script to download and preview the Kaggle insurance dataset."""

from __future__ import annotations

import csv
from pathlib import Path

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - lint aid
    import kagglehub as kagglehub_module  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]


def load_optional_module(module_name: str, pip_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - setup check
        raise SystemExit(
            f"Missing optional dependency '{pip_name}'. "
            f"Install it with `pip install {pip_name}` and retry."
        ) from exc


def format_table(rows: list[list[str]]) -> str:
    """Return a simple table string for a list of rows."""
    if not rows:
        return ""

    # Determine maximum width for each column.
    col_widths = [max(len(value) for value in column) for column in zip(*rows)]

    # Build the formatted lines.
    lines: list[str] = []
    for idx, row in enumerate(rows):
        line = " | ".join(value.ljust(col_width) for value, col_width in zip(row, col_widths))
        lines.append(line)
        if idx == 0:
            separator = "-+-".join("-" * col_width for col_width in col_widths)
            lines.append(separator)

    return "\n".join(lines)


def preview_csv(csv_path: Path, limit: int = 10) -> str:
    """Read the CSV and return up to `limit` rows formatted as a table."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            return "CSV file is empty."

        rows = [header]
        total_rows = 0
        for total_rows, row in enumerate(reader, start=1):
            if total_rows <= limit:
                rows.append([str(value) for value in row])

    table = format_table(rows)
    previewed = min(total_rows, limit)
    return f"Previewing first {previewed} of {total_rows} rows:\n\n{table}"


def create_age_charge_plot(csv_path: Path, output_path: Path | None = None) -> Path:
    pyplot = load_optional_module("matplotlib.pyplot", "matplotlib")

    ages: list[float] = []
    charges: list[float] = []
    colors: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                ages.append(float(row["age"]))
                charges.append(float(row["charges"]))
            except (TypeError, ValueError, KeyError):
                # Skip rows with missing or invalid numeric values.
                continue
            smoker = row.get("smoker", "unknown")
            colors.append("tab:red" if smoker == "yes" else "tab:blue")

    if not ages:
        raise ValueError("No valid rows found to plot.")

    pyplot.figure(figsize=(8, 6))
    pyplot.scatter(ages, charges, c=colors, alpha=0.7, edgecolors="k", linewidths=0.5)
    pyplot.title("Insurance Charges vs. Age")
    pyplot.xlabel("Age")
    pyplot.ylabel("Charges")
    handles = [
        pyplot.Line2D([], [], marker="o", linestyle="", color="tab:red", label="Smoker"),
        pyplot.Line2D([], [], marker="o", linestyle="", color="tab:blue", label="Non-Smoker"),
    ]
    pyplot.legend(handles=handles, loc="upper left")
    pyplot.tight_layout()

    target = output_path or (Path.cwd() / "insurance_age_charges.png")
    pyplot.savefig(target, dpi=150)
    pyplot.close()
    return target


def main() -> None:
    kagglehub = load_optional_module("kagglehub", "kagglehub")
    path = kagglehub.dataset_download("mirichoi0218/insurance")
    print(f"Path to dataset files: {path}")

    csv_path = Path(path) / "insurance.csv"
    if not csv_path.exists():
        available = ", ".join(sorted(p.name for p in Path(path).iterdir()))
        print("Could not find insurance.csv in the dataset directory.")
        print(f"Available files: {available or 'None'}")
        return

    print(preview_csv(csv_path))

    try:
        plot_path = create_age_charge_plot(csv_path)
    except ValueError as err:
        print(f"Could not generate plot: {err}")
    else:
        print(f"Saved scatter plot to: {plot_path}")


if __name__ == "__main__":
    main()

